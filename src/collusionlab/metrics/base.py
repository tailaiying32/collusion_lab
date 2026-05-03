"""Metrics infrastructure: RunData, LogReader, MetricsComputer ABC + registry.

Operates purely on saved ``log.jsonl`` + ``manifest.json`` — no import of
runner or environment packages required. Equilibrium anchors are resolved
from the manifest config (primary source) with an optional fallback to
per-round observations if present.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from collusionlab.storage import get_run_store, parse_db_run_ref, parse_db_sweep_ref

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RunData
# ---------------------------------------------------------------------------


@dataclass
class RunData:
    """Structured representation of a single experiment run.

    Built from ``manifest.json`` + ``log.jsonl``.  Provides convenient typed
    access without coupling to the runner or environment packages.
    """

    run_id: str
    env_type: str
    config: dict[str, Any]
    manifest: dict[str, Any]
    rounds: list[dict[str, Any]]

    # Pre-extracted series for fast access
    actions: list[list]  = field(default_factory=list)
    rewards: list[list[float]] = field(default_factory=list)
    messages: list[list[dict]] = field(default_factory=list)
    audit_events: list[dict | None] = field(default_factory=list)
    trajectory_signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    observations: list[dict] = field(default_factory=list)

    # Equilibrium anchors (resolved from manifest config)
    nash_price: float | None = None
    monopoly_price: float | None = None
    nash_profit: float | None = None
    monopoly_profit: float | None = None

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)

    @property
    def n_agents(self) -> int:
        env = self.config.get("environment", {})
        return env.get("n_agents", 2)

    @property
    def communication_mode(self) -> str:
        return self.config.get("communication_mode", "none")

    @property
    def oversight_mode(self) -> str:
        ov = self.config.get("oversight", {})
        return ov.get("mode", "none")

    @property
    def memory_window(self) -> int | None:
        agents = self.config.get("agents", {})
        if isinstance(agents, dict):
            return agents.get("memory_window")
        if isinstance(agents, list) and agents:
            return agents[0].get("memory_window")
        return None

    @property
    def seed(self) -> int | None:
        return self.config.get("environment", {}).get("seed")


# ---------------------------------------------------------------------------
# LogReader
# ---------------------------------------------------------------------------


class LogReader:
    """Load experiment outputs into ``RunData`` objects."""

    @staticmethod
    def load_run(manifest_path: str | Path) -> RunData:
        """Load a single run from its ``manifest.json`` path."""
        db_ref = parse_db_run_ref(manifest_path)
        if db_ref is not None:
            uri, run_id = db_ref
            store = get_run_store(uri)
            manifest = store.load_manifest(run_id)
            if manifest is None:
                raise FileNotFoundError(f"run {run_id!r} not found in {uri!r}")
            rounds = store.load_rounds(run_id)
            return LogReader._build_run_data(manifest, rounds)

        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        log_path = manifest_path.parent / "log.jsonl"
        rounds: list[dict] = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rounds.append(json.loads(line))

        return LogReader._build_run_data(manifest, rounds)

    @staticmethod
    def _build_run_data(manifest: dict[str, Any], rounds: list[dict]) -> RunData:
        config = manifest.get("config", {})
        env_cfg = config.get("environment", {})

        actions = [r.get("actions", []) for r in rounds]
        rewards = [r.get("rewards", []) for r in rounds]
        messages = [r.get("messages", []) for r in rounds]
        audit_events = [r.get("audit_event") for r in rounds]
        observations = [r.get("observations", {}) for r in rounds]

        sig_records: list[dict] = []
        for r in rounds:
            sig = dict(r.get("trajectory_signals", {}))
            sig["round"] = r.get("round", 0)
            sig_records.append(sig)
        sig_df = pd.DataFrame(sig_records)
        if "round" in sig_df.columns:
            sig_df = sig_df.set_index("round")

        return RunData(
            run_id=manifest.get("run_id", ""),
            env_type=manifest.get("env_type", config.get("env_type", "")),
            config=config,
            manifest=manifest,
            rounds=rounds,
            actions=actions,
            rewards=rewards,
            messages=messages,
            audit_events=audit_events,
            trajectory_signals=sig_df,
            observations=observations,
            nash_price=env_cfg.get("nash_price"),
            monopoly_price=env_cfg.get("monopoly_price"),
        )

    @classmethod
    def load_sweep(cls, sweep_manifest_path: str | Path) -> list[RunData]:
        """Load all runs referenced by a ``sweep_manifest.json``."""
        db_ref = parse_db_sweep_ref(sweep_manifest_path)
        if db_ref is not None:
            uri, sweep_id = db_ref
            sweep = get_run_store(uri).load_sweep_manifest(sweep_id)
            if sweep is None:
                raise FileNotFoundError(f"sweep {sweep_id!r} not found in {uri!r}")
            return cls._load_sweep_runs(sweep)

        sweep_manifest_path = Path(sweep_manifest_path)
        sweep = json.loads(sweep_manifest_path.read_text(encoding="utf-8"))
        return cls._load_sweep_runs(sweep)

    @classmethod
    def _load_sweep_runs(cls, sweep: dict[str, Any]) -> list[RunData]:
        runs: list[RunData] = []
        for entry in sweep.get("runs", []):
            if entry.get("status") != "succeeded":
                continue
            mp = entry.get("manifest_path")
            if mp is None:
                continue
            try:
                runs.append(cls.load_run(mp))
            except Exception as exc:
                logger.warning(
                    "Failed to load run %s: %s", entry.get("run_id"), exc
                )
        return runs


# ---------------------------------------------------------------------------
# MetricsComputer ABC + registry
# ---------------------------------------------------------------------------


class MetricsComputer(ABC):
    """Abstract base for per-env_type metrics computation.

    Subclasses compose generic and environment-specific metrics into a
    unified dict (single run) or DataFrame (sweep).
    """

    @abstractmethod
    def compute(self, run: RunData) -> dict[str, Any]:
        """Compute all metrics for a single run."""

    def compute_sweep(self, runs: list[RunData]) -> pd.DataFrame:
        """Compute metrics for every run and return one row per run."""
        rows = []
        for run in runs:
            row = self.compute(run)
            row.setdefault("run_id", run.run_id)
            row.setdefault("env_type", run.env_type)
            row.setdefault("seed", run.seed)
            row.setdefault("communication_mode", run.communication_mode)
            row.setdefault("oversight_mode", run.oversight_mode)
            row.setdefault("n_agents", run.n_agents)
            row.setdefault("memory_window", run.memory_window)
            rows.append(row)
        return pd.DataFrame(rows)


_METRICS_REGISTRY: dict[str, type[MetricsComputer]] = {}


def register_metrics(env_type: str, cls: type[MetricsComputer]) -> None:
    _METRICS_REGISTRY[env_type] = cls


def get_metrics_computer(env_type: str) -> MetricsComputer:
    if env_type not in _METRICS_REGISTRY:
        raise KeyError(
            f"no MetricsComputer registered for {env_type!r}; "
            f"registered: {sorted(_METRICS_REGISTRY)}"
        )
    return _METRICS_REGISTRY[env_type]()
