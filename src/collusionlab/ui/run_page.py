"""Run page — configure and launch a single experiment with live progress.

The launch button kicks off `Experiment(cfg).run()` in a daemon thread. The
thread mutates a plain dict held in `st.session_state`; the main script polls
that dict via a `time.sleep + st.rerun` loop while the run is in flight. No
Streamlit APIs are called from the worker thread, so no script-context attach
is needed.
"""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from collusionlab.runner.config import ExperimentConfig
from collusionlab.runner.experiment import Experiment

# Load .env once at import so OPENAI_API_KEY is visible to the worker thread.
load_dotenv()

CONFIGS_DIR = Path("configs")
STATE_KEY = "run_page_state"
EDITOR_KEY = "run_page_yaml_editor"
FILE_SELECT_KEY = "run_page_file_select"
CUSTOM_LABEL = "(custom)"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _default_state() -> dict[str, Any]:
    return {
        "running": False,
        "current_round": 0,
        "total_rounds": 0,
        "log_lines": [],
        "manifest_path": None,
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


def _list_config_files() -> list[Path]:
    if not CONFIGS_DIR.exists():
        return []
    return sorted(CONFIGS_DIR.glob("*.yaml"))


def _validate_yaml(text: str) -> tuple[ExperimentConfig | None, str | None]:
    if not text.strip():
        return None, "Editor is empty."
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return None, f"YAML parse error: {e}"
    if not isinstance(data, dict):
        return None, "Top-level YAML must be a mapping."
    if isinstance(data.get("environment"), dict):
        data["environment"].pop("_calibration_note", None)
    try:
        return ExperimentConfig(**data), None
    except ValidationError as e:
        return None, f"Validation error:\n{e}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


def _run_worker(cfg: ExperimentConfig, state: dict[str, Any]) -> None:
    def progress(round_idx: int, total: int, line: dict) -> None:
        state["current_round"] = round_idx
        state["total_rounds"] = total
        state["log_lines"].append(line)

    try:
        manifest_path = Experiment(cfg, progress_callback=progress).run()
        state["manifest_path"] = str(manifest_path)
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"
        state["traceback"] = traceback.format_exc()
    finally:
        state["running"] = False


def _start_run(cfg: ExperimentConfig) -> None:
    state = _ensure_state()
    state.clear()
    state.update(_default_state())
    state["running"] = True
    state["total_rounds"] = cfg.environment.n_rounds
    state["started_at"] = time.time()
    thread = threading.Thread(target=_run_worker, args=(cfg, state), daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------


def _init_editor_state() -> None:
    files = _list_config_files()
    file_labels = [CUSTOM_LABEL] + [p.name for p in files]
    if FILE_SELECT_KEY not in st.session_state:
        st.session_state[FILE_SELECT_KEY] = (
            "base.yaml" if "base.yaml" in file_labels else file_labels[0]
        )
    if EDITOR_KEY not in st.session_state:
        sel = st.session_state[FILE_SELECT_KEY]
        if sel != CUSTOM_LABEL:
            st.session_state[EDITOR_KEY] = (CONFIGS_DIR / sel).read_text(encoding="utf-8")
        else:
            st.session_state[EDITOR_KEY] = ""


def _on_file_change() -> None:
    sel = st.session_state[FILE_SELECT_KEY]
    if sel != CUSTOM_LABEL:
        st.session_state[EDITOR_KEY] = (CONFIGS_DIR / sel).read_text(encoding="utf-8")


def _render_editor() -> str:
    files = _list_config_files()
    file_labels = [CUSTOM_LABEL] + [p.name for p in files]
    st.selectbox(
        "Load config",
        options=file_labels,
        key=FILE_SELECT_KEY,
        on_change=_on_file_change,
        help="Pick a YAML from configs/. Editing switches to (custom).",
    )
    return st.text_area(
        "Config YAML",
        height=420,
        key=EDITOR_KEY,
    )


# ---------------------------------------------------------------------------
# Live rendering
# ---------------------------------------------------------------------------


def _render_live_chart(log_lines: list[dict], cfg_for_refs: dict | None = None) -> None:
    if not log_lines:
        return
    rounds = [l["round"] for l in log_lines]
    n_agents = len(log_lines[0].get("actions", []))
    fig = go.Figure()
    for i in range(n_agents):
        ys = [l["actions"][i] for l in log_lines]
        fig.add_trace(go.Scatter(x=rounds, y=ys, mode="lines+markers", name=f"Agent {i}"))
    if cfg_for_refs:
        nash = cfg_for_refs.get("nash_price")
        monopoly = cfg_for_refs.get("monopoly_price")
        if nash is not None:
            fig.add_hline(y=nash, line_dash="dash", line_color="green",
                          annotation_text="Nash", annotation_position="bottom right")
        if monopoly is not None:
            fig.add_hline(y=monopoly, line_dash="dash", line_color="red",
                          annotation_text="Monopoly", annotation_position="top right")
    fig.update_layout(xaxis_title="Round", yaxis_title="Action", height=320)
    st.plotly_chart(fig, width="stretch")


def _render_live_metrics(log_lines: list[dict], state: dict, cumulative_cost: float | None = None) -> None:
    last = log_lines[-1]
    elapsed = time.time() - (state.get("started_at") or time.time())
    rewards = last.get("rewards", [])
    cum = last.get("cumulative_rewards", [])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Round", f"{last['round']} / {state['total_rounds']}")
    col2.metric("Elapsed (s)", f"{elapsed:.1f}")
    col3.metric("Latest rewards", ", ".join(f"{r:.2f}" for r in rewards) or "—")
    col4.metric("Cumulative", ", ".join(f"{c:.1f}" for c in cum) or "—")


def _render_latest_reasoning(log_lines: list[dict]) -> None:
    last = log_lines[-1]
    reasoning = last.get("reasoning") or []
    if not any(r for r in reasoning):
        return
    with st.expander(f"Latest internal reasoning (round {last['round']})", expanded=False):
        st.caption("Private to each agent — not visible to other agents, the game, or the auditor.")
        for i, text in enumerate(reasoning):
            st.markdown(f"**Agent {i}:**")
            if text:
                st.markdown(f"> {text}")
            else:
                st.markdown("_No reasoning captured — fallback action._")


def _render_running(state: dict, env_cfg: dict | None) -> None:
    cur = state.get("current_round", 0)
    total = max(state.get("total_rounds", 1), 1)
    frac = min(cur / total, 1.0)
    st.progress(frac, text=f"Round {cur} / {total}")

    log_lines = list(state.get("log_lines", []))
    if log_lines:
        _render_live_metrics(log_lines, state)
        _render_live_chart(log_lines, env_cfg)
        _render_latest_reasoning(log_lines)
    else:
        st.caption("Waiting for first round to complete…")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def render_run_page() -> None:
    st.header("Run Experiment")
    state = _ensure_state()
    _init_editor_state()

    # Validate the current editor contents (used for both env_cfg refs while
    # running and the launch button gate).
    cfg, err = _validate_yaml(st.session_state.get(EDITOR_KEY, ""))
    env_cfg = (
        cfg.environment.model_dump() if cfg is not None else None
    )

    # --- Running: show live progress and self-rerun every second ----------
    if state["running"]:
        st.info("Experiment running…")
        _render_running(state, env_cfg)
        # Poll-and-rerun pattern. The fragment-based alternative would scope
        # less of the page, but this is simpler and the editor is hidden here.
        time.sleep(1.0)
        st.rerun()
        return

    # --- Completed run banner --------------------------------------------
    if state["manifest_path"]:
        st.success(f"Run complete — manifest at `{state['manifest_path']}`")
        log_lines = state.get("log_lines", [])
        if log_lines:
            _render_live_metrics(log_lines, state)
            _render_live_chart(log_lines, env_cfg)
            _render_latest_reasoning(log_lines)
        col1, col2 = st.columns([1, 4])
        if col1.button("View in Analyze tab"):
            st.session_state["nav_page"] = "Analyze"
            st.rerun()
        if col2.button("Clear and configure another run"):
            st.session_state[STATE_KEY] = _default_state()
            st.rerun()
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

    # --- Editor + launch --------------------------------------------------
    _render_editor()

    if err:
        st.warning(err)
    else:
        st.success(
            f"Config valid — env `{cfg.env_type}`, "
            f"{cfg.environment.n_agents} agents, "
            f"{cfg.environment.n_rounds} rounds, "
            f"comm `{cfg.communication_mode}`, "
            f"oversight `{cfg.oversight.mode}`."
        )

    if st.button("Launch experiment", type="primary", disabled=cfg is None):
        _start_run(cfg)
        st.rerun()
