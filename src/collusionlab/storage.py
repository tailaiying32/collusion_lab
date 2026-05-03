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
from typing import Any, Protocol
from urllib.parse import quote, unquote, urlparse


STORAGE_ENV_VAR = "COLLUSIONLAB_STORAGE_URL"


def configured_storage_uri(explicit_uri: str | None = None) -> str | None:
    """Return an explicit or environment-provided storage URI."""
    return explicit_uri or os.getenv(STORAGE_ENV_VAR)


def is_sqlite_uri(uri: str | None) -> bool:
    if not uri:
        return False
    return uri.startswith("sqlite://") or uri.endswith(".db") or uri.endswith(".sqlite")


def is_postgres_uri(uri: str | None) -> bool:
    if not uri:
        return False
    return uri.startswith("postgresql://") or uri.startswith("postgres://")


def is_database_uri(uri: str | None) -> bool:
    return is_sqlite_uri(uri) or is_postgres_uri(uri)


class RunStore(Protocol):
    uri: str

    def save_manifest(
        self,
        manifest: dict[str, Any],
        status: str = "succeeded",
        error: str | None = None,
    ) -> None: ...

    def append_round(self, row: dict[str, Any]) -> None: ...

    def load_manifest(self, run_id: str) -> dict[str, Any] | None: ...

    def load_rounds(self, run_id: str) -> list[dict[str, Any]]: ...

    def list_runs(self) -> list[dict[str, Any]]: ...

    def save_sweep_manifest(
        self,
        manifest: dict[str, Any],
        status: str = "succeeded",
        error: str | None = None,
    ) -> None: ...

    def load_sweep_manifest(self, sweep_id: str) -> dict[str, Any] | None: ...

    def list_sweeps(self) -> list[dict[str, Any]]: ...


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


def make_db_sweep_ref(uri: str, sweep_id: str) -> str:
    return (
        f"collusionlab-db://sweep?uri={quote(uri, safe='')}"
        f"&sweep_id={quote(sweep_id, safe='')}"
    )


def parse_db_run_ref(ref: str | Path) -> tuple[str, str] | None:
    raw = str(ref)
    if not raw.startswith("collusionlab-db://run?"):
        return None
    return _parse_db_ref_query(raw, "run_id")


def parse_db_sweep_ref(ref: str | Path) -> tuple[str, str] | None:
    raw = str(ref)
    if not raw.startswith("collusionlab-db://sweep?"):
        return None
    return _parse_db_ref_query(raw, "sweep_id")


def _parse_db_ref_query(raw: str, id_key: str) -> tuple[str, str] | None:
    query = raw.split("?", 1)[1]
    parts: dict[str, str] = {}
    for chunk in query.split("&"):
        if "=" in chunk:
            key, value = chunk.split("=", 1)
            parts[key] = unquote(value)
    uri = parts.get("uri")
    item_id = parts.get(id_key)
    if not uri or not item_id:
        return None
    return uri, item_id


def run_metadata_from_manifest(
    manifest: dict[str, Any],
    run_ref: str | Path,
) -> dict[str, Any]:
    config = manifest.get("config", {})
    agents_cfg_raw = config.get("agents", {})
    if isinstance(agents_cfg_raw, dict):
        agents_cfg = agents_cfg_raw
    elif (
        isinstance(agents_cfg_raw, list)
        and agents_cfg_raw
        and isinstance(agents_cfg_raw[0], dict)
    ):
        agents_cfg = agents_cfg_raw[0]
    else:
        agents_cfg = {}
    env_cfg = config.get("environment", {})
    oversight_cfg = config.get("oversight", {})
    return {
        "run_id": manifest.get("run_id", ""),
        "run_dir": run_ref,
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
    }


def sweep_metadata_from_manifest(
    manifest: dict[str, Any],
    sweep_ref: str | Path,
) -> dict[str, Any]:
    runs = manifest.get("runs", []) or []
    n_succeeded = sum(1 for run in runs if run.get("status") == "succeeded")
    n_failed = sum(1 for run in runs if run.get("status") != "succeeded")
    return {
        "sweep_id": manifest.get("sweep_id", ""),
        "sweep_dir": sweep_ref,
        "path": sweep_ref,
        "started_at": manifest.get("started_at", ""),
        "ended_at": manifest.get("ended_at"),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "mode": manifest.get("mode", "unknown"),
        "base_config": manifest.get("base_config"),
        "max_workers": manifest.get("max_workers"),
        "n_runs": len(runs),
        "n_succeeded": n_succeeded,
        "n_failed": n_failed,
        "status": manifest.get("status", "unknown"),
    }


def get_run_store(uri: str) -> RunStore:
    if is_postgres_uri(uri):
        return PostgresRunStore(uri)
    if is_sqlite_uri(uri):
        return SQLiteRunStore(uri)
    raise ValueError(f"unsupported storage URI: {uri!r}")


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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sweeps (
                    sweep_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT,
                    mode TEXT,
                    base_config TEXT,
                    max_workers INTEGER,
                    n_runs INTEGER,
                    n_succeeded INTEGER,
                    n_failed INTEGER,
                    manifest_json TEXT NOT NULL,
                    error TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save_manifest(
        self,
        manifest: dict[str, Any],
        status: str = "succeeded",
        error: str | None = None,
    ) -> None:
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
            records.append(
                run_metadata_from_manifest(
                    {**manifest, "run_id": manifest.get("run_id", run_id)},
                    make_db_run_ref(self.uri, str(run_id)),
                )
            )
        return records

    def save_sweep_manifest(
        self,
        manifest: dict[str, Any],
        status: str = "succeeded",
        error: str | None = None,
    ) -> None:
        sweep_id = str(manifest["sweep_id"])
        payload = json.dumps({**manifest, "status": status}, sort_keys=True)
        runs = manifest.get("runs", []) or []
        n_succeeded = sum(1 for run in runs if run.get("status") == "succeeded")
        n_failed = sum(1 for run in runs if run.get("status") != "succeeded")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sweeps (
                    sweep_id, status, started_at, ended_at, mode, base_config,
                    max_workers, n_runs, n_succeeded, n_failed, manifest_json,
                    error, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(sweep_id) DO UPDATE SET
                    status=excluded.status,
                    started_at=excluded.started_at,
                    ended_at=excluded.ended_at,
                    mode=excluded.mode,
                    base_config=excluded.base_config,
                    max_workers=excluded.max_workers,
                    n_runs=excluded.n_runs,
                    n_succeeded=excluded.n_succeeded,
                    n_failed=excluded.n_failed,
                    manifest_json=excluded.manifest_json,
                    error=excluded.error,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    sweep_id,
                    status,
                    manifest.get("started_at"),
                    manifest.get("ended_at"),
                    manifest.get("mode"),
                    manifest.get("base_config"),
                    manifest.get("max_workers"),
                    len(runs),
                    n_succeeded,
                    n_failed,
                    payload,
                    error,
                ),
            )

    def load_sweep_manifest(self, sweep_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT manifest_json FROM sweeps WHERE sweep_id = ?", (sweep_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list_sweeps(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT sweep_id, manifest_json FROM sweeps
                WHERE status IN ('running', 'succeeded')
                ORDER BY COALESCE(started_at, '') DESC
                """
            ).fetchall()
        records: list[dict[str, Any]] = []
        for sweep_id, manifest_json in rows:
            manifest = json.loads(manifest_json)
            records.append(
                sweep_metadata_from_manifest(
                    {**manifest, "sweep_id": manifest.get("sweep_id", sweep_id)},
                    make_db_sweep_ref(self.uri, str(sweep_id)),
                )
            )
        return records


class PostgresRunStore:
    """Postgres-backed storage for run manifests and round JSON rows."""

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self._init_schema()

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "Postgres storage requires the 'psycopg[binary]' package. "
                "Install/update the collusion_lab environment from environment.yml."
            ) from exc
        return psycopg.connect(self.uri, autocommit=True, row_factory=dict_row)

    @staticmethod
    def _jsonb(value: dict[str, Any]):
        from psycopg.types.json import Jsonb

        return Jsonb(value)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    env_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TIMESTAMPTZ,
                    end_time TIMESTAMPTZ,
                    communication_mode TEXT,
                    oversight_mode TEXT,
                    firm_backend TEXT,
                    firm_model TEXT,
                    n_rounds INTEGER,
                    n_agents INTEGER,
                    manifest_json JSONB NOT NULL,
                    error TEXT,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    updated_at TIMESTAMPTZ DEFAULT now()
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS round_logs (
                    run_id TEXT NOT NULL,
                    round INTEGER NOT NULL,
                    row_json JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    updated_at TIMESTAMPTZ DEFAULT now(),
                    PRIMARY KEY (run_id, round),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sweeps (
                    sweep_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    started_at TIMESTAMPTZ,
                    ended_at TIMESTAMPTZ,
                    mode TEXT,
                    base_config TEXT,
                    max_workers INTEGER,
                    n_runs INTEGER,
                    n_succeeded INTEGER,
                    n_failed INTEGER,
                    manifest_json JSONB NOT NULL,
                    error TEXT,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    updated_at TIMESTAMPTZ DEFAULT now()
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_start_time ON runs (start_time DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_env_type ON runs (env_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_round_logs_run_round ON round_logs (run_id, round)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sweeps_started_at ON sweeps (started_at DESC)"
            )

    def save_manifest(
        self,
        manifest: dict[str, Any],
        status: str = "succeeded",
        error: str | None = None,
    ) -> None:
        run_id = str(manifest["run_id"])
        metadata = run_metadata_from_manifest(manifest, make_db_run_ref(self.uri, run_id))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, env_type, status, start_time, end_time,
                    communication_mode, oversight_mode, firm_backend, firm_model,
                    n_rounds, n_agents, manifest_json, error, updated_at
                )
                VALUES (
                    %(run_id)s, %(env_type)s, %(status)s, %(start_time)s, %(end_time)s,
                    %(communication_mode)s, %(oversight_mode)s, %(firm_backend)s,
                    %(firm_model)s, %(n_rounds)s, %(n_agents)s, %(manifest_json)s,
                    %(error)s, now()
                )
                ON CONFLICT(run_id) DO UPDATE SET
                    env_type=excluded.env_type,
                    status=excluded.status,
                    start_time=excluded.start_time,
                    end_time=excluded.end_time,
                    communication_mode=excluded.communication_mode,
                    oversight_mode=excluded.oversight_mode,
                    firm_backend=excluded.firm_backend,
                    firm_model=excluded.firm_model,
                    n_rounds=excluded.n_rounds,
                    n_agents=excluded.n_agents,
                    manifest_json=excluded.manifest_json,
                    error=excluded.error,
                    updated_at=now()
                """,
                {
                    "run_id": run_id,
                    "env_type": str(metadata.get("env_type") or ""),
                    "status": status,
                    "start_time": manifest.get("start_time"),
                    "end_time": manifest.get("end_time"),
                    "communication_mode": metadata.get("comm_mode"),
                    "oversight_mode": metadata.get("oversight_mode"),
                    "firm_backend": metadata.get("firm_backend"),
                    "firm_model": metadata.get("firm_model"),
                    "n_rounds": metadata.get("n_rounds"),
                    "n_agents": metadata.get("n_agents"),
                    "manifest_json": self._jsonb(manifest),
                    "error": error,
                },
            )

    def append_round(self, row: dict[str, Any]) -> None:
        run_id = str(row["run_id"])
        round_num = int(row["round"])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO round_logs (run_id, round, row_json, updated_at)
                VALUES (%(run_id)s, %(round)s, %(row_json)s, now())
                ON CONFLICT(run_id, round) DO UPDATE SET
                    row_json=excluded.row_json,
                    updated_at=now()
                """,
                {
                    "run_id": run_id,
                    "round": round_num,
                    "row_json": self._jsonb(row),
                },
            )

    def load_manifest(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT manifest_json FROM runs WHERE run_id = %s", (run_id,)
            ).fetchone()
        return row["manifest_json"] if row else None

    def load_rounds(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT row_json FROM round_logs
                WHERE run_id = %s
                ORDER BY round ASC
                """,
                (run_id,),
            ).fetchall()
        return [row["row_json"] for row in rows]

    def list_runs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, manifest_json FROM runs
                WHERE status IN ('running', 'succeeded')
                ORDER BY start_time DESC NULLS LAST
                """
            ).fetchall()
        return [
            run_metadata_from_manifest(
                {**row["manifest_json"], "run_id": row["manifest_json"].get("run_id", row["run_id"])},
                make_db_run_ref(self.uri, str(row["run_id"])),
            )
            for row in rows
        ]

    def save_sweep_manifest(
        self,
        manifest: dict[str, Any],
        status: str = "succeeded",
        error: str | None = None,
    ) -> None:
        sweep_id = str(manifest["sweep_id"])
        manifest_with_status = {**manifest, "status": status}
        runs = manifest.get("runs", []) or []
        n_succeeded = sum(1 for run in runs if run.get("status") == "succeeded")
        n_failed = sum(1 for run in runs if run.get("status") != "succeeded")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sweeps (
                    sweep_id, status, started_at, ended_at, mode, base_config,
                    max_workers, n_runs, n_succeeded, n_failed, manifest_json,
                    error, updated_at
                )
                VALUES (
                    %(sweep_id)s, %(status)s, %(started_at)s, %(ended_at)s,
                    %(mode)s, %(base_config)s, %(max_workers)s, %(n_runs)s,
                    %(n_succeeded)s, %(n_failed)s, %(manifest_json)s,
                    %(error)s, now()
                )
                ON CONFLICT(sweep_id) DO UPDATE SET
                    status=excluded.status,
                    started_at=excluded.started_at,
                    ended_at=excluded.ended_at,
                    mode=excluded.mode,
                    base_config=excluded.base_config,
                    max_workers=excluded.max_workers,
                    n_runs=excluded.n_runs,
                    n_succeeded=excluded.n_succeeded,
                    n_failed=excluded.n_failed,
                    manifest_json=excluded.manifest_json,
                    error=excluded.error,
                    updated_at=now()
                """,
                {
                    "sweep_id": sweep_id,
                    "status": status,
                    "started_at": manifest.get("started_at"),
                    "ended_at": manifest.get("ended_at"),
                    "mode": manifest.get("mode"),
                    "base_config": manifest.get("base_config"),
                    "max_workers": manifest.get("max_workers"),
                    "n_runs": len(runs),
                    "n_succeeded": n_succeeded,
                    "n_failed": n_failed,
                    "manifest_json": self._jsonb(manifest_with_status),
                    "error": error,
                },
            )

    def load_sweep_manifest(self, sweep_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT manifest_json FROM sweeps WHERE sweep_id = %s", (sweep_id,)
            ).fetchone()
        return row["manifest_json"] if row else None

    def list_sweeps(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT sweep_id, manifest_json FROM sweeps
                WHERE status IN ('running', 'succeeded')
                ORDER BY started_at DESC NULLS LAST
                """
            ).fetchall()
        return [
            sweep_metadata_from_manifest(
                {**row["manifest_json"], "sweep_id": row["manifest_json"].get("sweep_id", row["sweep_id"])},
                make_db_sweep_ref(self.uri, str(row["sweep_id"])),
            )
            for row in rows
        ]
