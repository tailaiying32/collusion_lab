"""Data loading utilities for the UI.

Provides functions to discover runs, load manifests/logs, and extract
trajectory data for plotting. Uses Streamlit caching where appropriate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def list_runs(raw_dir: Path | str) -> list[dict]:
    """Scan raw_dir for run directories and return metadata sorted by start_time (newest first).

    Each entry contains:
        - run_id: str
        - run_dir: Path
        - start_time: str (ISO format)
        - env_type: str
        - comm_mode: str
        - oversight_mode: str
    """
    raw_dir = Path(raw_dir)
    runs: list[dict] = []

    if not raw_dir.exists():
        return runs

    for manifest_path in raw_dir.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = manifest.get("config", {})
            runs.append({
                "run_id": manifest.get("run_id", manifest_path.parent.name),
                "run_dir": manifest_path.parent,
                "start_time": manifest.get("start_time", ""),
                "env_type": manifest.get("env_type", config.get("env_type", "unknown")),
                "comm_mode": config.get("communication_mode", "unknown"),
                "oversight_mode": config.get("oversight", {}).get("mode", "unknown"),
            })
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read manifest at %s: %s", manifest_path, e)
            continue

    runs.sort(key=lambda r: r["start_time"], reverse=True)
    return runs


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_manifest(run_dir: Path | str) -> dict | None:
    """Load a run's manifest.json. Returns None on failure."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load manifest at %s: %s", manifest_path, e)
        return None


def _load_manifest_cached(run_dir: str) -> dict | None:
    """Streamlit-cached wrapper around load_manifest."""
    return load_manifest(run_dir)


if _HAS_STREAMLIT:
    _load_manifest_cached = st.cache_data(_load_manifest_cached)


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------


def load_log_rows(run_dir: Path | str) -> list[dict]:
    """Load a run's log.jsonl as a list of round dicts. Returns empty list on failure."""
    run_dir = Path(run_dir)
    log_path = run_dir / "log.jsonl"
    if not log_path.exists():
        return []
    rows: list[dict] = []
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load log at %s: %s", log_path, e)
        return []
    return rows


def _load_log_rows_cached(run_dir: str) -> list[dict]:
    """Streamlit-cached wrapper around load_log_rows."""
    return load_log_rows(run_dir)


if _HAS_STREAMLIT:
    _load_log_rows_cached = st.cache_data(_load_log_rows_cached)


# ---------------------------------------------------------------------------
# Trajectory extraction
# ---------------------------------------------------------------------------


def extract_trajectory_df(rows: list[dict]) -> pd.DataFrame:
    """Flatten round logs into a DataFrame for plotting.

    Columns:
        - round: int
        - action_0, action_1, ...: per-agent actions
        - reward_0, reward_1, ...: per-agent rewards
        - reward_elevation_0, reward_elevation_1, ...: per-agent elevation
        - action_spread: float
        - explicit_collusion_flag, behavior_collusion_flag, covert_coordination_flag, hollow_coordination_flag: bool
    """
    records: list[dict[str, Any]] = []

    for row in rows:
        rec: dict[str, Any] = {"round": row.get("round", 0)}

        actions = row.get("actions", [])
        for i, a in enumerate(actions):
            rec[f"action_{i}"] = a

        rewards = row.get("rewards", [])
        for i, r in enumerate(rewards):
            rec[f"reward_{i}"] = r

        signals = row.get("trajectory_signals", {})

        elev = signals.get("reward_elevation", [])
        for i, e in enumerate(elev):
            rec[f"reward_elevation_{i}"] = e

        rec["action_spread"] = signals.get("action_spread")
        rec["explicit_collusion_flag"] = signals.get("explicit_collusion_flag", False)
        rec["behavior_collusion_flag"] = signals.get("behavior_collusion_flag", False)
        rec["covert_coordination_flag"] = signals.get("covert_coordination_flag", False)
        rec["hollow_coordination_flag"] = signals.get("hollow_coordination_flag", False)

        records.append(rec)

    return pd.DataFrame(records)
