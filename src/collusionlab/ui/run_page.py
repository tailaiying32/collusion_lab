"""Run page — configure and launch a single experiment with live progress.

The launch button kicks off `Experiment(cfg).run()` in a daemon thread. The
thread mutates a plain dict held in `st.session_state`; the main script polls
that dict via a `time.sleep + st.rerun` loop while the run is in flight. No
Streamlit APIs are called from the worker thread, so no script-context attach
is needed.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from collusionlab.runner.config import ExperimentConfig
from collusionlab.runner.experiment import Experiment
from collusionlab.ui.data_loading import get_recent_config, set_recent_config

# Load .env once at import so OPENAI_API_KEY is visible to the worker thread.
load_dotenv()

CONFIGS_DIR = Path("configs")
STATE_KEY = "run_page_state"
EDITOR_KEY = "run_page_yaml_editor"
FILE_SELECT_KEY = "run_page_file_select"
CUSTOM_LABEL = "(custom)"
RECENT_RUN_CONFIG_KEY = "run_page_last_config"
BACKEND_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-7"],
    "deepseek": ["deepseek-v4-flash"],
}
MODEL_TO_BACKEND: dict[str, str] = {
    model: backend
    for backend, models in BACKEND_MODELS.items()
    for model in models
}


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
    # Backward compatibility: collapse identical list-style agent blocks into
    # one shared config.
    agents = data.get("agents")
    if isinstance(agents, list) and agents:
        first = agents[0]
        if all(a == first for a in agents[1:]):
            data["agents"] = first
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
        recent = get_recent_config(RECENT_RUN_CONFIG_KEY)
        if recent in file_labels:
            st.session_state[FILE_SELECT_KEY] = recent
        else:
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
        set_recent_config(RECENT_RUN_CONFIG_KEY, sel)
        st.session_state[EDITOR_KEY] = (CONFIGS_DIR / sel).read_text(encoding="utf-8")


def _build_selector_options(file_names: list[str], recent_name: str | None) -> list[str]:
    options = [CUSTOM_LABEL]
    for name in file_names:
        if recent_name and name == recent_name:
            options.append(f"{name} (most recent)")
        else:
            options.append(name)
    return options


def _selector_value_to_name(value: str) -> str:
    return value.replace(" (most recent)", "")


def _patch_agent_models(text: str, updates: dict[str, str]) -> str:
    """Patch YAML editor text with selected model/backend values."""
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return text
    if not isinstance(data, dict):
        return text
    agents = data.get("agents")
    if not isinstance(agents, dict):
        return text
    agents.update(updates)
    data["agents"] = agents
    return yaml.safe_dump(data, sort_keys=False)


def _patch_auditor_models(text: str, updates: dict[str, str]) -> str:
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return text
    if not isinstance(data, dict):
        return text
    oversight = data.get("oversight")
    if not isinstance(oversight, dict):
        return text
    oversight.update(updates)
    data["oversight"] = oversight
    return yaml.safe_dump(data, sort_keys=False)


def _render_model_controls(cfg: ExperimentConfig | None) -> None:
    if cfg is None:
        return
    model_options = list(MODEL_TO_BACKEND.keys())
    if cfg.agents.model not in model_options:
        model_options = [cfg.agents.model] + model_options
    selected_model = st.selectbox(
        "Firm model",
        options=model_options,
        index=model_options.index(cfg.agents.model),
    )
    selected_backend = MODEL_TO_BACKEND.get(selected_model, cfg.agents.backend)
    if selected_model != cfg.agents.model or selected_backend != cfg.agents.backend:
        st.session_state[EDITOR_KEY] = _patch_agent_models(
            st.session_state.get(EDITOR_KEY, ""),
            {"backend": selected_backend, "model": selected_model},
        )
        st.rerun()

    oversight = cfg.oversight
    if oversight.mode != "audit-penalty" or not oversight.llm_judge_enabled:
        return
    backends = sorted(BACKEND_MODELS.keys())
    judge_backend = oversight.llm_judge_backend
    if judge_backend not in backends:
        backends.append(judge_backend)
    judge_models = list(BACKEND_MODELS.get(judge_backend, []))
    if oversight.llm_judge_model not in judge_models:
        judge_models = [oversight.llm_judge_model] + judge_models
    c3, c4 = st.columns(2)
    selected_judge_backend = c3.selectbox(
        "Auditor backend",
        options=backends,
        index=backends.index(judge_backend),
    )
    selected_judge_model_options = list(BACKEND_MODELS.get(selected_judge_backend, []))
    if not selected_judge_model_options:
        selected_judge_model_options = [oversight.llm_judge_model]
    if oversight.llm_judge_model not in selected_judge_model_options:
        selected_judge_model_options = [oversight.llm_judge_model] + selected_judge_model_options
    selected_judge_model = c4.selectbox(
        "Auditor model",
        options=selected_judge_model_options,
        index=(
            selected_judge_model_options.index(oversight.llm_judge_model)
            if oversight.llm_judge_model in selected_judge_model_options
            else 0
        ),
    )
    if (
        selected_judge_backend != judge_backend
        or selected_judge_model != oversight.llm_judge_model
    ):
        st.session_state[EDITOR_KEY] = _patch_auditor_models(
            st.session_state.get(EDITOR_KEY, ""),
            {
                "llm_judge_backend": selected_judge_backend,
                "llm_judge_model": selected_judge_model,
            },
        )
        st.rerun()


def _render_editor() -> str:
    files = _list_config_files()
    file_names = [p.name for p in files]
    recent_name = get_recent_config(RECENT_RUN_CONFIG_KEY)
    file_labels = _build_selector_options(file_names, recent_name)
    selected_name = st.session_state.get(FILE_SELECT_KEY, CUSTOM_LABEL)
    default_label = (
        f"{selected_name} (most recent)"
        if recent_name and selected_name == recent_name and selected_name != CUSTOM_LABEL
        else selected_name
    )
    if default_label not in file_labels:
        default_label = CUSTOM_LABEL
    st.selectbox(
        "Load config",
        options=file_labels,
        key=f"{FILE_SELECT_KEY}_display",
        index=file_labels.index(default_label),
        help="Pick a YAML from configs/. Editing switches to (custom).",
    )
    selected_display = st.session_state.get(f"{FILE_SELECT_KEY}_display", CUSTOM_LABEL)
    normalized = _selector_value_to_name(selected_display)
    if st.session_state.get(FILE_SELECT_KEY) != normalized:
        st.session_state[FILE_SELECT_KEY] = normalized
        _on_file_change()
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
    if cumulative_cost is not None:
        st.caption(f"Token/cost estimate from completed manifest: ${cumulative_cost:.4f}")
    else:
        st.caption("Token and cost totals are finalized when the manifest is written at run completion.")


def _get_llm_judge_verdict(audit_event: dict | None) -> str:
    if not audit_event:
        return "—"
    for r in audit_event.get("results", []) or []:
        if r.get("auditor") == "llm_judge":
            d = r.get("details", {}) or {}
            if d.get("skipped"):
                return "—"
            return d.get("verdict") or "—"
    return "—"


def _render_recent_rounds(log_lines: list[dict], n: int = 5) -> None:
    if not log_lines:
        return
    rows = []
    for line in log_lines[-n:]:
        signals = line.get("trajectory_signals", {})
        elev = signals.get("reward_elevation", [])
        rows.append({
            "round": line.get("round"),
            "actions": line.get("actions"),
            "rewards": [round(r, 3) for r in line.get("rewards", [])],
            "mean_elevation": round(sum(elev) / len(elev), 3) if elev else None,
            "spread": signals.get("action_spread"),
            "judge": _get_llm_judge_verdict(line.get("audit_event")),
            "covert": signals.get("covert_coordination_flag", False),
            "hollow": signals.get("hollow_coordination_flag", False),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


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


def _render_latest_audit(log_lines: list[dict]) -> None:
    last = log_lines[-1]
    audit = last.get("audit_event")
    if not audit:
        return
    with st.expander(f"Latest audit (round {last['round']})", expanded=False):
        for result in audit.get("results", []) or []:
            name = result.get("auditor")
            details = result.get("details", {}) or {}
            if name == "llm_judge":
                if details.get("skipped"):
                    st.caption("LLM judge: skipped (no messages this round).")
                    continue
                verdict = details.get("verdict") or "—"
                st.markdown(f"**LLM judge:** {verdict}")
                if details.get("evidence"):
                    st.markdown(f"> {details['evidence']}")
                if details.get("reasoning"):
                    st.caption(details["reasoning"])
                if details.get("error"):
                    st.warning(f"Judge call failed: {details['error']}")
            elif name == "behavior":
                elev = details.get("current_elevation")
                spread = details.get("current_spread")
                sustained = details.get("sustained_rounds")
                above = details.get("above_threshold")
                st.caption(
                    f"Behavior: elevation={elev}, spread={spread}, "
                    f"sustained_rounds={sustained}, above_threshold={above}"
                )


def _render_running(state: dict, env_cfg: dict | None) -> None:
    cur = state.get("current_round", 0)
    total = max(state.get("total_rounds", 1), 1)
    frac = min(cur / total, 1.0)
    st.progress(frac, text=f"Round {cur} / {total}")

    log_lines = list(state.get("log_lines", []))
    if log_lines:
        _render_live_metrics(log_lines, state)
        _render_live_chart(log_lines, env_cfg)
        st.subheader("Recent Rounds")
        _render_recent_rounds(log_lines)
        _render_latest_reasoning(log_lines)
        _render_latest_audit(log_lines)
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
        cost = None
        run_id = None
        try:
            manifest = json.loads(Path(state["manifest_path"]).read_text(encoding="utf-8"))
            cost = manifest.get("total_cost_estimate_usd")
            run_id = manifest.get("run_id")
        except Exception:
            pass
        if log_lines:
            _render_live_metrics(log_lines, state, cumulative_cost=cost)
            _render_live_chart(log_lines, env_cfg)
            st.subheader("Recent Rounds")
            _render_recent_rounds(log_lines)
            _render_latest_reasoning(log_lines)
            _render_latest_audit(log_lines)
        col1, col2 = st.columns([1, 4])
        if col1.button("View in Analyze tab"):
            if run_id:
                st.session_state["_selected_run_id"] = run_id
            st.session_state["_nav_target"] = "Analyze"
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
    _render_model_controls(cfg)
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
        selected = st.session_state.get(FILE_SELECT_KEY, CUSTOM_LABEL)
        if selected != CUSTOM_LABEL:
            set_recent_config(RECENT_RUN_CONFIG_KEY, selected)
        _start_run(cfg)
        st.rerun()
