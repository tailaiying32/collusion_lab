"""Persistence backends for experiment run artifacts.

The file layout remains the default compatibility backend.  The SQLite backend
adds a database-backed copy of manifests and round logs so a UI process can read
results without sharing the experiment host's output directory.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse


STORAGE_ENV_VAR = "COLLUSIONLAB_STORAGE_URL"


def configured_storage_uri(explicit_uri: str | None = None) -> str | None:
    """Return an explicit or environment-provided storage URI."""
    return explicit_uri or os.getenv(STORAGE_ENV_VAR)


def is_sqlite_uri(uri: str | None) -> bool:
    if not uri:
        return False
    return uri.startswith("sqlite://") or uri.endswith(".db") or uri.endswith(".sqlite")


def sqlite_path_from_uri(uri: str) -> Path | str:
    """Parse sqlite:///path/to/db.sqlite or a plain filesystem path."""
    if uri == "sqlite:///:memory:":
        return ":memory:"
    if uri.startswith("sqlite:///"):
        return Path(uri.removeprefix("sqlite:///")).expanduser()
    if uri.startswith("sqlite://"):
        parsed = urlparse(uri)
        path = parsed.path or parsed.netloc
        return Path(path).expanduser()
    return Path(uri).expanduser()


def make_db_run_ref(uri: str, run_id: str) -> str:
    return f"collusionlab-db://run?uri={quote(uri, safe='')}&run_id={quote(run_id, safe='')}"


def parse_db_run_ref(ref: str | Path) -> tuple[str, str] | None:
    raw = str(ref)
    if not raw.startswith("collusionlab-db://run?"):
        return None
    query = raw.split("?", 1)[1]
    parts: dict[str, str] = {}
    for chunk in query.split("&"):
        if "=" in chunk:
            key, value = chunk.split("=", 1)
            parts[key] = unquote(value)
    uri = parts.get("uri")
    run_id = parts.get("run_id")
    if not uri or not run_id:
        return None
    return uri, run_id


class SQLiteRunStore:
    """SQLite-backed storage for run manifests and round JSON rows."""

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.path = sqlite_path_from_uri(uri)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    env_type TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    status TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS round_logs (
                    run_id TEXT NOT NULL,
                    round INTEGER NOT NULL,
                    row_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, round),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                )
                """
            )

    def save_manifest(self, manifest: dict[str, Any], status: str = "succeeded") -> None:
        run_id = str(manifest["run_id"])
        payload = json.dumps(manifest, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, env_type, start_time, end_time, status, manifest_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(run_id) DO UPDATE SET
                    env_type=excluded.env_type,
                    start_time=excluded.start_time,
                    end_time=excluded.end_time,
                    status=excluded.status,
                    manifest_json=excluded.manifest_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    run_id,
                    str(manifest.get("env_type", "")),
                    manifest.get("start_time"),
                    manifest.get("end_time"),
                    status,
                    payload,
                ),
            )

    def append_round(self, row: dict[str, Any]) -> None:
        run_id = str(row["run_id"])
        round_num = int(row["round"])
        payload = json.dumps(row, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO round_logs (run_id, round, row_json)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id, round) DO UPDATE SET row_json=excluded.row_json
                """,
                (run_id, round_num, payload),
            )

    def load_manifest(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT manifest_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def load_rounds(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT row_json FROM round_logs
                WHERE run_id = ?
                ORDER BY round ASC
                """,
                (run_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def list_runs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, manifest_json FROM runs
                WHERE status IN ('running', 'succeeded')
                ORDER BY COALESCE(start_time, '') DESC
                """
            ).fetchall()
        records: list[dict[str, Any]] = []
        for run_id, manifest_json in rows:
            manifest = json.loads(manifest_json)
            config = manifest.get("config", {})
            agents_cfg_raw = config.get("agents", {})
            if isinstance(agents_cfg_raw, dict):
                agents_cfg = agents_cfg_raw
            elif isinstance(agents_cfg_raw, list) and agents_cfg_raw and isinstance(agents_cfg_raw[0], dict):
                agents_cfg = agents_cfg_raw[0]
            else:
                agents_cfg = {}
            env_cfg = config.get("environment", {})
            oversight_cfg = config.get("oversight", {})
            records.append({
                "run_id": manifest.get("run_id", run_id),
                "run_dir": make_db_run_ref(self.uri, str(run_id)),
                "start_time": manifest.get("start_time", ""),
                "env_type": manifest.get("env_type", config.get("env_type", "unknown")),
                "comm_mode": config.get("communication_mode", "unknown"),
                "oversight_mode": oversight_cfg.get("mode", "unknown"),
                "n_rounds": env_cfg.get("n_rounds"),
                "n_agents": env_cfg.get("n_agents"),
                "firm_backend": agents_cfg.get("backend"),
                "firm_model": agents_cfg.get("model"),
                "memory_window": agents_cfg.get("memory_window"),
                "audit_probability": oversight_cfg.get("audit_probability"),
                "auditor_model": oversight_cfg.get("llm_judge_model"),
            })
        return records
