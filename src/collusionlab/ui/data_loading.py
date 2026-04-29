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


def list_sweeps(raw_dir: Path | str) -> list[dict]:
    """Scan raw_dir for sweep directories and return metadata (newest first)."""
    raw_dir = Path(raw_dir)
    sweeps: list[dict] = []
    if not raw_dir.exists():
        return sweeps

    for sweep_manifest_path in raw_dir.glob("sweep_*/sweep_manifest.json"):
        try:
            sweep = json.loads(sweep_manifest_path.read_text(encoding="utf-8"))
            sweeps.append({
                "sweep_id": sweep.get("sweep_id", sweep_manifest_path.parent.name),
                "sweep_dir": sweep_manifest_path.parent,
                "path": sweep_manifest_path,
                "started_at": sweep.get("started_at", ""),
                "mode": sweep.get("mode", "unknown"),
                "n_runs": len(sweep.get("runs", [])),
            })
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read sweep manifest at %s: %s", sweep_manifest_path, e)
            continue

    sweeps.sort(key=lambda s: s["started_at"], reverse=True)
    return sweeps


def load_sweep_manifest(path: Path | str) -> dict | None:
    """Load sweep_manifest.json. Returns None on failure."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load sweep manifest at %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Schema-safe adapters
# ---------------------------------------------------------------------------


def get_signal(row: dict, key: str, default: Any = None) -> Any:
    """Read a trajectory signal with backward-compatible fallbacks."""
    signals = row.get("trajectory_signals", {}) or {}
    if key in signals:
        return signals[key]
    if key == "price_follow_indicator":
        return signals.get("price_follow_lag1", default)
    return default


def build_run_index(raw_dir: Path | str) -> pd.DataFrame:
    """Return normalized run selector metadata as a DataFrame."""
    runs = list_runs(raw_dir)
    records: list[dict[str, Any]] = []
    for r in runs:
        started = pd.to_datetime(r.get("start_time"), errors="coerce")
        records.append({
            **r,
            "date": started.date() if not pd.isna(started) else None,
            "label": (
                f"run_id={r.get('run_id', '')[:8]}... | "
                f"time={str(r.get('start_time', ''))[:19]} | "
                f"env={r.get('env_type', 'unknown')} | "
                f"comm={r.get('comm_mode', 'unknown')} | "
                f"oversight={r.get('oversight_mode', 'unknown')}"
            ),
        })
    return pd.DataFrame(records)


def build_transcript_df(
    rows: list[dict],
    onset_round: int | None = None,
    transition_round: int | None = None,
) -> pd.DataFrame:
    """Flatten transcript rows with derived filter columns."""
    records: list[dict[str, Any]] = []
    for row in rows:
        round_num = row.get("round", 0)
        audit_event = row.get("audit_event")
        messages = row.get("messages", []) or []
        records.append({
            "round": round_num,
            "actions": row.get("actions", []),
            "rewards": row.get("rewards", []),
            "messages": messages,
            "message_text": "\n".join(m.get("content", "") for m in messages),
            "audit_event": audit_event,
            "audited": bool(audit_event and audit_event.get("audited")),
            "flagged": bool(audit_event and audit_event.get("flagged")),
            "penalized": bool(audit_event and audit_event.get("penalty_applied")),
            "explicit": bool(get_signal(row, "explicit_collusion_flag", False)),
            "behavior": bool(get_signal(row, "behavior_collusion_flag", False)),
            "covert": bool(get_signal(row, "covert_coordination_flag", False)),
            "hollow": bool(get_signal(row, "hollow_coordination_flag", False)),
            "price_follow_indicator": get_signal(row, "price_follow_indicator"),
            "post_onset": onset_round is not None and round_num >= onset_round,
            "post_transition": transition_round is not None and round_num >= transition_round,
            "raw": row,
        })
    return pd.DataFrame(records)


def build_compare_df(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize sweep metrics for the Compare run browser."""
    if sweep_df.empty:
        return sweep_df.copy()
    df = sweep_df.copy()
    if "mean_covert_rate" not in df.columns and "covert_flag_count" in df.columns:
        denom = df.get("n_rounds", 1).replace(0, 1)
        df["mean_covert_rate"] = df["covert_flag_count"] / denom
    if "mean_hollow_rate" not in df.columns and "hollow_flag_count" in df.columns:
        denom = df.get("n_rounds", 1).replace(0, 1)
        df["mean_hollow_rate"] = df["hollow_flag_count"] / denom
    if "concealment_gap" not in df.columns:
        covert = df.get("mean_covert_rate", 0)
        hollow = df.get("mean_hollow_rate", 0)
        df["concealment_gap"] = covert - hollow
    if "has_onset" not in df.columns and "onset_round" in df.columns:
        df["has_onset"] = df["onset_round"].notna()
    if "has_transition" not in df.columns and "transition_round" in df.columns:
        df["has_transition"] = df["transition_round"].notna()
    return df


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
