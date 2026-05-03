"""Sweep runner — parallel execution of experiment grids.

SweepConfig specifies a base YAML plus overrides in grid or list mode.
SweepRunner expands configs, runs them in parallel via ProcessPoolExecutor,
and writes a sweep_manifest.json with per-run status/timing/errors.

CLI:
    python -m collusionlab.runner.sweep --sweep configs/sweep_stego_study.yaml --max-workers 4
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field, model_validator

from collusionlab.storage import (
    STORAGE_ENV_VAR,
    configured_storage_uri,
    get_run_store,
    is_database_uri,
    is_postgres_uri,
    make_db_run_ref,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dot-path override helpers
# ---------------------------------------------------------------------------


def _resolve_path(data: dict | list, path: str) -> tuple[Any, str | int]:
    """Walk a dot-path and return ``(parent_container, final_key)``.

    Raises :class:`ValueError` on unknown keys or out-of-range indices.
    """
    parts = path.split(".")
    current: Any = data

    for i, part in enumerate(parts[:-1]):
        traversed = ".".join(parts[: i + 1])
        if part.isdigit():
            idx = int(part)
            if not isinstance(current, list):
                raise ValueError(
                    f"Override path {path!r}: segment {part!r} is an index "
                    f"but value at {traversed!r} is {type(current).__name__}, not list"
                )
            if idx < 0 or idx >= len(current):
                raise ValueError(
                    f"Override path {path!r}: index {idx} out of range "
                    f"(list at {'.'.join(parts[:i]) or '<root>'!r} has "
                    f"{len(current)} elements)"
                )
            current = current[idx]
        else:
            if not isinstance(current, dict):
                raise ValueError(
                    f"Override path {path!r}: expected dict at "
                    f"{traversed!r}, got {type(current).__name__}"
                )
            if part not in current:
                raise ValueError(
                    f"Override path {path!r}: key {part!r} not found "
                    f"(available: {sorted(current.keys())})"
                )
            current = current[part]

    final = parts[-1]
    if final.isdigit():
        idx = int(final)
        if not isinstance(current, list):
            raise ValueError(
                f"Override path {path!r}: final segment is an index "
                f"but target is {type(current).__name__}, not list"
            )
        if idx < 0 or idx >= len(current):
            raise ValueError(
                f"Override path {path!r}: index {idx} out of range "
                f"(list has {len(current)} elements)"
            )
        return current, idx

    if not isinstance(current, dict):
        raise ValueError(
            f"Override path {path!r}: expected dict for final segment, "
            f"got {type(current).__name__}"
        )
    if final not in current:
        raise ValueError(
            f"Override path {path!r}: key {final!r} not found "
            f"(available: {sorted(current.keys())})"
        )
    return current, final


def _check_type_compat(path: str, original: Any, override: Any) -> None:
    """Raise :class:`ValueError` if the override type is incompatible."""
    if original is None or override is None:
        return

    orig_type = type(original)
    over_type = type(override)

    # int ↔ float are interchangeable for numeric config fields.
    if {orig_type, over_type} <= {int, float}:
        return

    if orig_type is not over_type:
        raise ValueError(
            f"Override path {path!r}: type mismatch — base config has "
            f"{orig_type.__name__} ({original!r}), override provides "
            f"{over_type.__name__} ({override!r})"
        )


def apply_overrides(data: dict, overrides: dict[str, Any]) -> dict:
    """Deep-copy *data* and apply strict dot-path overrides.

    Raises :class:`ValueError` on unknown paths or type mismatches.
    """
    result = copy.deepcopy(data)
    for path in sorted(overrides):
        value = overrides[path]
        container, key = _resolve_path(result, path)
        _check_type_compat(path, container[key], value)
        container[key] = value
    return result


# ---------------------------------------------------------------------------
# SweepConfig
# ---------------------------------------------------------------------------


class SweepConfig(BaseModel):
    """Schema for a parameter sweep definition YAML."""

    base_config: str
    mode: Literal["grid", "list"] = "grid"
    # Grid: dict mapping dot-paths → list of values (Cartesian product).
    # List: list of dicts, each mapping dot-paths → scalar values.
    overrides: dict[str, list[Any]] | list[dict[str, Any]]

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_overrides_shape(self) -> "SweepConfig":
        if self.mode == "grid":
            if not isinstance(self.overrides, dict):
                raise ValueError(
                    "Grid mode requires 'overrides' to be a dict mapping "
                    "dot-paths to lists of values"
                )
            for key, vals in self.overrides.items():
                if not isinstance(vals, list) or len(vals) == 0:
                    raise ValueError(
                        f"Grid override {key!r} must be a non-empty list"
                    )
        else:
            if not isinstance(self.overrides, list):
                raise ValueError(
                    "List mode requires 'overrides' to be a list of dicts"
                )
            if len(self.overrides) == 0:
                raise ValueError(
                    "List mode requires at least one override dict"
                )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SweepConfig":
        with Path(path).open() as f:
            data = yaml.safe_load(f)
        return cls(**data)

    # ------------------------------------------------------------------
    # Config expansion
    # ------------------------------------------------------------------

    def expand(self) -> list[dict]:
        """Generate resolved config dicts from *base_config* + *overrides*.

        Each returned dict is a complete ``ExperimentConfig``-compatible dict
        with a unique ``run_id``.  Expansion order is deterministic for a given
        sweep definition.
        """
        base_path = Path(self.base_config)
        with base_path.open() as f:
            base_data = yaml.safe_load(f)

        if isinstance(base_data.get("environment"), dict):
            base_data["environment"].pop("_calibration_note", None)

        if self.mode == "grid":
            return self._expand_grid(base_data)
        return self._expand_list(base_data)

    def _expand_grid(self, base_data: dict) -> list[dict]:
        assert isinstance(self.overrides, dict)
        keys = sorted(self.overrides.keys())
        value_lists = [self.overrides[k] for k in keys]

        configs: list[dict] = []
        for combo in itertools.product(*value_lists):
            overrides_dict = dict(zip(keys, combo))
            config = apply_overrides(base_data, overrides_dict)
            config["run_id"] = str(uuid.uuid4())
            configs.append(config)
        return configs

    def _expand_list(self, base_data: dict) -> list[dict]:
        assert isinstance(self.overrides, list)
        configs: list[dict] = []
        for entry in self.overrides:
            config = apply_overrides(base_data, entry)
            config["run_id"] = str(uuid.uuid4())
            configs.append(config)
        return configs


# ---------------------------------------------------------------------------
# CLI storage preflight + progress display
# ---------------------------------------------------------------------------


def summarize_sweep_storage(
    configs: list[dict],
    *,
    verify_connection: bool = True,
) -> str:
    """Validate and summarize where sweep results will be persisted."""
    uri = resolve_sweep_storage_uri(configs)
    if not configs:
        return "Storage: no runs in sweep."

    if uri is None:
        return "Storage: local only. Results will not be persisted to Neon/Postgres."
    if verify_connection:
        # Constructor initializes schema; list_runs verifies a basic read path.
        get_run_store(uri).list_runs()

    storage_entries = [cfg.get("storage") or {"backend": "local", "uri": None} for cfg in configs]
    source = "storage.uri" if any(entry.get("uri") for entry in storage_entries) else STORAGE_ENV_VAR
    if is_postgres_uri(uri):
        return (
            f"Storage: postgres via {source} ({_describe_storage_uri(uri)}). "
            "Sweep, runs, and round logs will be persisted to DB plus local output_dir."
        )
    return (
        f"Storage: database via {source} ({_describe_storage_uri(uri)}). "
        "Sweep, runs, and round logs will be persisted to DB plus local output_dir."
    )


def resolve_sweep_storage_uri(configs: list[dict]) -> str | None:
    """Resolve the single database URI used by all configs in a sweep."""
    storage_entries = [cfg.get("storage") or {"backend": "local", "uri": None} for cfg in configs]
    resolved_uris: set[str] = set()

    for entry in storage_entries:
        backend = str(entry.get("backend", "local"))
        explicit_uri = entry.get("uri")
        uri = (
            configured_storage_uri(explicit_uri)
            if backend != "local" or explicit_uri
            else None
        )
        if backend != "local" or is_database_uri(uri):
            if not uri:
                raise ValueError(
                    "database storage is enabled, but no storage URI is configured. "
                    f"Set {STORAGE_ENV_VAR} on the HPC job or provide storage.uri."
                )
            resolved_uris.add(uri)
            if backend == "postgres" and not is_postgres_uri(uri):
                raise ValueError(
                    "storage.backend is 'postgres', but the resolved storage URI "
                    f"is not PostgreSQL: {_describe_storage_uri(uri)}"
                )

    if not resolved_uris:
        return None
    if len(resolved_uris) > 1:
        targets = ", ".join(sorted(_describe_storage_uri(uri) for uri in resolved_uris))
        raise ValueError(f"sweep resolves to multiple storage targets: {targets}")
    return next(iter(resolved_uris))


def db_persistable_sweep_manifest(manifest: dict[str, Any], storage_uri: str) -> dict[str, Any]:
    """Return a sweep manifest whose successful runs point at DB run refs."""
    db_manifest = copy.deepcopy(manifest)
    for run in db_manifest.get("runs", []) or []:
        run_id = run.get("run_id")
        manifest_path = run.get("manifest_path")
        if run_id and manifest_path:
            run["local_manifest_path"] = manifest_path
            run["manifest_path"] = make_db_run_ref(storage_uri, str(run_id))
    return db_manifest


def _describe_storage_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme in {"postgres", "postgresql"}:
        host = parsed.hostname or "unknown-host"
        database = parsed.path.lstrip("/") or "unknown-db"
        return f"{parsed.scheme}://{host}/{database}"
    if parsed.scheme:
        return f"{parsed.scheme}://{parsed.netloc or parsed.path}"
    return Path(uri).name or uri


class TerminalSweepProgress:
    def __init__(self, total: int, *, stream=None, width: int = 24) -> None:
        self.total = max(total, 1)
        self.stream = stream or sys.stdout
        self.width = width
        self.started_at = time.perf_counter()
        self.ok = 0
        self.failed = 0
        self._interactive = bool(getattr(self.stream, "isatty", lambda: False)())

    def update(self, result: dict) -> None:
        if result.get("status") == "succeeded":
            self.ok += 1
        else:
            self.failed += 1
        done = self.ok + self.failed
        line = format_sweep_progress(
            done=done,
            total=self.total,
            ok=self.ok,
            failed=self.failed,
            elapsed=time.perf_counter() - self.started_at,
            width=self.width,
        )
        end = "" if self._interactive and done < self.total else "\n"
        prefix = "\r" if self._interactive else ""
        self.stream.write(prefix + line + end)
        self.stream.flush()


def format_sweep_progress(
    *,
    done: int,
    total: int,
    ok: int,
    failed: int,
    elapsed: float,
    width: int = 24,
) -> str:
    total = max(total, 1)
    done = min(max(done, 0), total)
    filled = int(round(width * done / total))
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"Sweep progress [{bar}] {done}/{total} complete | "
        f"ok={ok} failed={failed} | elapsed={elapsed:.1f}s"
    )


# ---------------------------------------------------------------------------
# Spawn-safe worker (top-level function for ProcessPoolExecutor on Windows)
# ---------------------------------------------------------------------------


def _init_worker_path(parent_sys_path: list[str]) -> None:
    """Propagate the parent's sys.path into spawned worker processes.

    Needed when the package is not pip-installed and the parent relies on
    ``sys.path.insert`` or ``PYTHONPATH`` to find ``collusionlab``.
    """
    import sys

    for p in reversed(parent_sys_path):
        if p not in sys.path:
            sys.path.insert(0, p)


def _run_single_experiment(config_data: dict) -> dict:
    """Execute one experiment inside a worker process.

    All imports are performed inside the function so that each spawned
    process triggers its own registry population.
    """
    import time as _time
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    run_id = config_data.get("run_id", "unknown")
    started_at = _dt.now(_tz.utc)
    start_perf = _time.perf_counter()

    try:
        # Side-effect-import to populate environment and backend registries.
        import collusionlab.environments  # noqa: F401
        from collusionlab.runner.config import ExperimentConfig
        from collusionlab.runner.experiment import Experiment

        cfg = ExperimentConfig(**config_data)
        manifest_path = Experiment(cfg).run()

        return {
            "run_id": run_id,
            "config": config_data,
            "status": "succeeded",
            "manifest_path": str(manifest_path),
            "error": None,
            "started_at": started_at.isoformat(),
            "ended_at": _dt.now(_tz.utc).isoformat(),
            "elapsed_seconds": _time.perf_counter() - start_perf,
        }
    except Exception as exc:
        return {
            "run_id": run_id,
            "config": config_data,
            "status": "failed",
            "manifest_path": None,
            "error": f"{type(exc).__name__}: {exc}",
            "started_at": started_at.isoformat(),
            "ended_at": _dt.now(_tz.utc).isoformat(),
            "elapsed_seconds": _time.perf_counter() - start_perf,
        }


# ---------------------------------------------------------------------------
# SweepRunner
# ---------------------------------------------------------------------------

SweepProgressCallback = Callable[[int, int], None]
SweepResultCallback = Callable[[dict], None]


class SweepRunner:
    """Expands a :class:`SweepConfig` and runs all experiments in parallel.

    Failed runs are recorded as ``status: "failed"`` in the sweep manifest
    (continue-on-error policy).
    """

    def __init__(
        self,
        sweep_config: SweepConfig,
        max_workers: int | None = None,
        output_dir: str | None = None,
        progress_callback: SweepProgressCallback | None = None,
        result_callback: SweepResultCallback | None = None,
    ) -> None:
        self.sweep_config = sweep_config
        self.max_workers = max_workers or os.cpu_count() or 1
        self.output_dir = output_dir
        self.progress_callback = progress_callback
        self.result_callback = result_callback

    def run(self) -> Path:
        """Execute the sweep and return the path to ``sweep_manifest.json``."""
        sweep_id = str(uuid.uuid4())
        configs = self.sweep_config.expand()

        if self.output_dir:
            for cfg in configs:
                cfg["output_dir"] = self.output_dir

        output_base = Path(
            configs[0]["output_dir"] if configs else "data/raw"
        )
        sweep_dir = output_base / f"sweep_{sweep_id}"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        sweep_manifest_path = sweep_dir / "sweep_manifest.json"

        started_at = datetime.now(timezone.utc)
        start_perf = time.perf_counter()
        n_total = len(configs)
        results: list[dict] = []
        storage_uri = resolve_sweep_storage_uri(configs)
        sweep_store = get_run_store(storage_uri) if storage_uri else None
        running_manifest = {
            "sweep_id": sweep_id,
            "status": "running",
            "started_at": started_at.isoformat(),
            "ended_at": None,
            "elapsed_seconds": None,
            "base_config": self.sweep_config.base_config,
            "mode": self.sweep_config.mode,
            "max_workers": self.max_workers,
            "runs": [],
        }
        sweep_manifest_path.write_text(
            json.dumps(running_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if sweep_store:
            sweep_store.save_sweep_manifest(
                db_persistable_sweep_manifest(running_manifest, storage_uri),
                status="running",
            )

        logger.info(
            "Starting sweep %s: %d runs, max_workers=%d",
            sweep_id,
            n_total,
            self.max_workers,
        )

        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_init_worker_path,
            initargs=(sys.path.copy(),),
        ) as executor:
            futures = {
                executor.submit(_run_single_experiment, cfg): cfg["run_id"]
                for cfg in configs
            }
            for future in as_completed(futures):
                run_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "run_id": run_id,
                        "config": None,
                        "status": "failed",
                        "manifest_path": None,
                        "error": f"Worker exception: {type(exc).__name__}: {exc}",
                        "started_at": None,
                        "ended_at": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": None,
                    }
                results.append(result)
                n_done = len(results)
                logger.info(
                    "[%d/%d] %s — %s (%.1fs)",
                    n_done,
                    n_total,
                    result["run_id"],
                    result["status"],
                    result.get("elapsed_seconds") or 0,
                )
                if self.progress_callback:
                    self.progress_callback(n_done, n_total)
                if self.result_callback:
                    self.result_callback(result)

        ended_at = datetime.now(timezone.utc)
        elapsed = time.perf_counter() - start_perf

        # Sort by run_id for deterministic manifest output regardless of
        # completion order.
        results.sort(key=lambda r: r["run_id"])

        manifest = {
            "sweep_id": sweep_id,
            "status": "succeeded",
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "elapsed_seconds": elapsed,
            "base_config": self.sweep_config.base_config,
            "mode": self.sweep_config.mode,
            "max_workers": self.max_workers,
            "runs": results,
        }
        sweep_manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if sweep_store:
            sweep_store.save_sweep_manifest(
                db_persistable_sweep_manifest(manifest, storage_uri),
                status="succeeded",
            )

        n_ok = sum(1 for r in results if r["status"] == "succeeded")
        n_fail = n_total - n_ok
        logger.info(
            "Sweep %s complete: %d succeeded, %d failed, %.1fs elapsed",
            sweep_id,
            n_ok,
            n_fail,
            elapsed,
        )
        return sweep_manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a CollusionLab parameter sweep.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sweep",
        dest="sweep_path",
        help="Path to SweepConfig YAML (canonical flag).",
    )
    group.add_argument(
        "--config",
        dest="sweep_path_alt",
        help="Alias for --sweep (mirrors single-run CLI).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel workers (default: CPU count).",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output_dir for all generated configs.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default INFO).",
    )
    parser.add_argument(
        "--skip-storage-preflight",
        action="store_true",
        help="Skip database storage connectivity check before launching workers.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )

    from dotenv import load_dotenv

    load_dotenv()

    sweep_path = args.sweep_path or args.sweep_path_alt
    sweep_cfg = SweepConfig.from_yaml(sweep_path)
    configs = sweep_cfg.expand()
    storage_summary = summarize_sweep_storage(
        configs,
        verify_connection=not args.skip_storage_preflight,
    )
    print(storage_summary)
    progress = TerminalSweepProgress(len(configs))
    runner = SweepRunner(
        sweep_config=sweep_cfg,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        result_callback=progress.update,
    )
    manifest_path = runner.run()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    n_ok = sum(1 for run in runs if run.get("status") == "succeeded")
    n_fail = len(runs) - n_ok
    print(
        f"Sweep complete: {n_ok} succeeded, {n_fail} failed, "
        f"{manifest.get('elapsed_seconds', 0):.1f}s elapsed"
    )
    if n_fail:
        print("Failed run summaries:")
        for run in [r for r in runs if r.get("status") != "succeeded"][:5]:
            print(f"- {run.get('run_id')}: {run.get('error')}")
    print(str(manifest_path))


if __name__ == "__main__":
    main()
