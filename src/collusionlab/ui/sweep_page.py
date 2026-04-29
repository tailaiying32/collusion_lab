"""Sweep page — configure, preview, launch, and monitor parameter sweeps.

The launch button kicks off ``SweepRunner`` in a daemon thread. The thread
mutates a plain dict held in ``st.session_state``; the main script polls
that dict via a ``time.sleep + st.rerun`` loop while the sweep is in flight.
No Streamlit APIs are called from the worker thread.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback as tb_mod
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from collusionlab.runner.sweep import SweepConfig, SweepRunner

load_dotenv()

CONFIGS_DIR = Path("configs")
STATE_KEY = "sweep_page_state"
EDITOR_KEY = "sweep_page_yaml_editor"
FILE_SELECT_KEY = "sweep_page_file_select"
WORKERS_KEY = "sweep_page_max_workers"
CUSTOM_LABEL = "(custom)"

PREVIEW_CAP = 50


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _default_state() -> dict[str, Any]:
    return {
        "running": False,
        "n_done": 0,
        "n_total": 0,
        "manifest_path": None,
        "runs": [],
        "error": None,
        "traceback": None,
        "started_at": None,
    }


def _ensure_state() -> dict[str, Any]:
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = _default_state()
    return st.session_state[STATE_KEY]


# ---------------------------------------------------------------------------
# Config discovery + validation
# ---------------------------------------------------------------------------


def _list_sweep_files() -> list[Path]:
    if not CONFIGS_DIR.exists():
        return []
    sweep_files = sorted(CONFIGS_DIR.glob("sweep_*.yaml"))
    other_files = sorted(
        p for p in CONFIGS_DIR.glob("*.yaml") if p not in set(sweep_files)
    )
    return sweep_files + other_files


def _validate_sweep_yaml(
    text: str,
) -> tuple[SweepConfig | None, str | None]:
    if not text.strip():
        return None, "Editor is empty."
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return None, f"YAML parse error: {e}"
    if not isinstance(data, dict):
        return None, "Top-level YAML must be a mapping."
    try:
        return SweepConfig(**data), None
    except ValidationError as e:
        return None, f"Validation error:\n{e}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _try_expand(
    sweep_cfg: SweepConfig,
) -> tuple[list[dict] | None, str | None]:
    """Attempt expansion; return configs or an error message."""
    try:
        configs = sweep_cfg.expand()
        return configs, None
    except ValueError as e:
        return None, f"Override error: {e}"
    except FileNotFoundError as e:
        return None, f"Base config not found: {e}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


def _sweep_worker(
    sweep_cfg: SweepConfig,
    max_workers: int,
    state: dict[str, Any],
) -> None:
    def on_progress(n_done: int, n_total: int) -> None:
        state["n_done"] = n_done
        state["n_total"] = n_total

    try:
        runner = SweepRunner(
            sweep_config=sweep_cfg,
            max_workers=max_workers,
            progress_callback=on_progress,
        )
        manifest_path = runner.run()
        state["manifest_path"] = str(manifest_path)
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        state["runs"] = manifest.get("runs", [])
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"
        state["traceback"] = tb_mod.format_exc()
    finally:
        state["running"] = False


def _start_sweep(sweep_cfg: SweepConfig, max_workers: int, n_total: int) -> None:
    state = _ensure_state()
    state.clear()
    state.update(_default_state())
    state["running"] = True
    state["n_total"] = n_total
    state["started_at"] = time.time()
    thread = threading.Thread(
        target=_sweep_worker,
        args=(sweep_cfg, max_workers, state),
        daemon=True,
    )
    thread.start()


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------


def _init_editor_state() -> None:
    files = _list_sweep_files()
    file_labels = [CUSTOM_LABEL] + [p.name for p in files]
    if FILE_SELECT_KEY not in st.session_state:
        default = next(
            (n for n in file_labels if n.startswith("sweep_")), file_labels[0]
        )
        st.session_state[FILE_SELECT_KEY] = default
    if EDITOR_KEY not in st.session_state:
        sel = st.session_state[FILE_SELECT_KEY]
        if sel != CUSTOM_LABEL:
            st.session_state[EDITOR_KEY] = (
                (CONFIGS_DIR / sel).read_text(encoding="utf-8")
            )
        else:
            st.session_state[EDITOR_KEY] = ""


def _on_file_change() -> None:
    sel = st.session_state[FILE_SELECT_KEY]
    if sel != CUSTOM_LABEL:
        st.session_state[EDITOR_KEY] = (
            (CONFIGS_DIR / sel).read_text(encoding="utf-8")
        )


def _render_editor() -> str:
    files = _list_sweep_files()
    file_labels = [CUSTOM_LABEL] + [p.name for p in files]
    st.selectbox(
        "Load sweep config",
        options=file_labels,
        key=FILE_SELECT_KEY,
        on_change=_on_file_change,
        help="Pick a sweep YAML from configs/. sweep_*.yaml files listed first.",
    )
    return st.text_area(
        "Sweep Config YAML",
        height=340,
        key=EDITOR_KEY,
    )


# ---------------------------------------------------------------------------
# Grid preview
# ---------------------------------------------------------------------------


def _render_grid_preview(
    sweep_cfg: SweepConfig, configs: list[dict],
) -> None:
    n = len(configs)
    st.markdown(f"**{n} run{'s' if n != 1 else ''}** will be generated.")

    if sweep_cfg.mode == "grid" and isinstance(sweep_cfg.overrides, dict):
        override_keys = sorted(sweep_cfg.overrides.keys())
    elif sweep_cfg.mode == "list" and isinstance(sweep_cfg.overrides, list):
        all_keys: set[str] = set()
        for entry in sweep_cfg.overrides:
            all_keys.update(entry.keys())
        override_keys = sorted(all_keys)
    else:
        override_keys = []

    if not override_keys:
        return

    rows: list[dict[str, Any]] = []
    display_configs = configs[:PREVIEW_CAP]
    for i, cfg in enumerate(display_configs):
        row: dict[str, Any] = {"#": i + 1}
        for key in override_keys:
            parts = key.split(".")
            val: Any = cfg
            for p in parts:
                if isinstance(val, list) and p.isdigit():
                    val = val[int(p)]
                elif isinstance(val, dict):
                    val = val.get(p, "—")
                else:
                    val = "—"
                    break
            row[key] = val
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if n > PREVIEW_CAP:
        st.caption(f"Showing {PREVIEW_CAP} of {n} runs.")


# ---------------------------------------------------------------------------
# Live progress
# ---------------------------------------------------------------------------


def _render_running(state: dict[str, Any]) -> None:
    n_done = state.get("n_done", 0)
    n_total = max(state.get("n_total", 1), 1)
    frac = min(n_done / n_total, 1.0)
    st.progress(frac, text=f"{n_done} / {n_total} runs complete")

    elapsed = time.time() - (state.get("started_at") or time.time())
    col1, col2, col3 = st.columns(3)
    col1.metric("Completed", f"{n_done} / {n_total}")
    col2.metric("Elapsed (s)", f"{elapsed:.1f}")
    if n_done > 0:
        eta = (elapsed / n_done) * (n_total - n_done)
        col3.metric("Est. remaining (s)", f"{eta:.0f}")
    else:
        col3.metric("Est. remaining (s)", "—")


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------


def _render_completed(state: dict[str, Any]) -> None:
    runs = state.get("runs", [])
    n_total = len(runs)
    n_ok = sum(1 for r in runs if r.get("status") == "succeeded")
    n_fail = n_total - n_ok
    elapsed = time.time() - (state.get("started_at") or time.time())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total runs", n_total)
    col2.metric("Succeeded", n_ok)
    col3.metric("Failed", n_fail)
    col4.metric("Elapsed (s)", f"{elapsed:.1f}")

    if runs:
        rows = []
        for r in runs:
            run_id = r.get("run_id", "?")
            rows.append({
                "run_id": run_id[:12] + "..." if len(run_id) > 12 else run_id,
                "status": r.get("status", "?"),
                "elapsed (s)": (
                    f"{r['elapsed_seconds']:.1f}"
                    if r.get("elapsed_seconds") is not None
                    else "—"
                ),
                "error": (r.get("error") or "")[:80],
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    col1, col2 = st.columns([1, 4])
    if col1.button("View in Compare tab"):
        st.session_state["_nav_target"] = "Compare"
        st.rerun()
    if col2.button("Clear and configure another sweep"):
        st.session_state[STATE_KEY] = _default_state()
        st.rerun()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def render_sweep_page() -> None:
    st.header("Parameter Sweep")
    state = _ensure_state()
    _init_editor_state()

    # --- Running: show live progress and self-rerun every second ----------
    if state["running"]:
        st.info("Sweep running...")
        _render_running(state)
        time.sleep(1.0)
        st.rerun()
        return

    # --- Completed sweep banner -------------------------------------------
    if state["manifest_path"]:
        st.success(
            f"Sweep complete — manifest at `{state['manifest_path']}`"
        )
        _render_completed(state)
        return

    # --- Error banner -----------------------------------------------------
    if state["error"]:
        st.error(state["error"])
        if state.get("traceback"):
            with st.expander("Traceback"):
                st.code(state["traceback"])
        if st.button("Dismiss error"):
            st.session_state[STATE_KEY] = _default_state()
            st.rerun()

    # --- Editor + validation + preview + launch ---------------------------
    _render_editor()

    sweep_cfg, err = _validate_sweep_yaml(
        st.session_state.get(EDITOR_KEY, "")
    )

    if err:
        st.warning(err)
        return

    st.success(
        f"Sweep config valid — mode `{sweep_cfg.mode}`, "
        f"base `{sweep_cfg.base_config}`."
    )

    configs, expand_err = _try_expand(sweep_cfg)
    if expand_err:
        st.warning(expand_err)
        return

    _render_grid_preview(sweep_cfg, configs)

    # Launch controls
    max_workers = st.number_input(
        "Max workers",
        min_value=1,
        max_value=32,
        value=min(os.cpu_count() or 1, 8),
        step=1,
        key=WORKERS_KEY,
        help="Number of parallel worker processes for the sweep.",
    )

    if st.button("Launch sweep", type="primary"):
        _start_sweep(sweep_cfg, int(max_workers), len(configs))
        st.rerun()
