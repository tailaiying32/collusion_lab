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
from hashlib import sha1
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from collusionlab.runner.config import ExperimentConfig
from collusionlab.runner.sweep import SweepConfig, SweepRunner
from collusionlab.ui.data_loading import get_recent_config, set_recent_config

load_dotenv()

CONFIGS_DIR = Path("configs")
STATE_KEY = "sweep_page_state"
EDITOR_KEY = "sweep_page_yaml_editor"
FILE_SELECT_KEY = "sweep_page_file_select"
WORKERS_KEY = "sweep_page_max_workers"
BASE_EDITOR_KEY = "sweep_page_base_yaml_editor"
BASE_PATH_KEY = "sweep_page_base_path"
BASE_FILE_SELECT_KEY = "sweep_page_base_file_select"
CUSTOM_LABEL = "(custom)"
RECENT_SWEEP_CONFIG_KEY = "sweep_page_last_sweep_config"
RECENT_BASE_CONFIG_KEY = "sweep_page_last_base_config"
BACKEND_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-5-mini", "gpt-5.1"],
    "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-7"],
    "deepseek": ["deepseek-v4-flash"],
}
MODEL_TO_BACKEND: dict[str, str] = {
    model: backend
    for backend, models in BACKEND_MODELS.items()
    for model in models
}
_COMM_OPTIONS = ["none", "public", "private"]
_AUDIT_OPTIONS = ["none", "audit-penalty"]

PREVIEW_CAP = 100


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


def _list_base_files() -> list[Path]:
    if not CONFIGS_DIR.exists():
        return []
    return sorted(CONFIGS_DIR.glob("*.yaml"))


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
        recent = get_recent_config(RECENT_SWEEP_CONFIG_KEY)
        default = (
            recent
            if recent in file_labels
            else next((n for n in file_labels if n.startswith("sweep_")), file_labels[0])
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
        set_recent_config(RECENT_SWEEP_CONFIG_KEY, sel)
        st.session_state[EDITOR_KEY] = (
            (CONFIGS_DIR / sel).read_text(encoding="utf-8")
        )


def _on_base_file_change() -> None:
    sel = st.session_state.get(BASE_FILE_SELECT_KEY, CUSTOM_LABEL)
    if sel != CUSTOM_LABEL:
        set_recent_config(RECENT_BASE_CONFIG_KEY, sel)
        path = CONFIGS_DIR / sel
        st.session_state[BASE_PATH_KEY] = str(path)
        st.session_state[BASE_EDITOR_KEY] = path.read_text(encoding="utf-8")


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


def _render_editor() -> str:
    files = _list_sweep_files()
    file_names = [p.name for p in files]
    recent_name = get_recent_config(RECENT_SWEEP_CONFIG_KEY)
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
        "Load sweep config",
        options=file_labels,
        key=f"{FILE_SELECT_KEY}_display",
        index=file_labels.index(default_label),
        help="Pick a sweep YAML from configs/. sweep_*.yaml files listed first.",
    )
    selected_display = st.session_state.get(f"{FILE_SELECT_KEY}_display", CUSTOM_LABEL)
    normalized = _selector_value_to_name(selected_display)
    if st.session_state.get(FILE_SELECT_KEY) != normalized:
        st.session_state[FILE_SELECT_KEY] = normalized
        _on_file_change()
    return st.text_area(
        "Sweep Config YAML",
        height=340,
        key=EDITOR_KEY,
    )


def _render_base_selector(default_path: Path | None) -> None:
    files = _list_base_files()
    file_names = [p.name for p in files]
    recent_name = get_recent_config(RECENT_BASE_CONFIG_KEY)
    file_labels = _build_selector_options(file_names, recent_name)
    if BASE_FILE_SELECT_KEY not in st.session_state:
        default_name = default_path.name if default_path and default_path.exists() else None
        st.session_state[BASE_FILE_SELECT_KEY] = (
            recent_name
            if recent_name in [CUSTOM_LABEL] + file_names
            else (
                default_name
                if default_name in [CUSTOM_LABEL] + file_names
                else (file_names[0] if file_names else CUSTOM_LABEL)
            )
        )
    selected_name = st.session_state.get(BASE_FILE_SELECT_KEY, CUSTOM_LABEL)
    default_label = (
        f"{selected_name} (most recent)"
        if recent_name and selected_name == recent_name and selected_name != CUSTOM_LABEL
        else selected_name
    )
    if default_label not in file_labels:
        default_label = CUSTOM_LABEL
    st.selectbox(
        "Load base config",
        options=file_labels,
        key=f"{BASE_FILE_SELECT_KEY}_display",
        index=file_labels.index(default_label),
        help="Pick a base experiment YAML from configs/ (or keep custom text).",
    )
    selected_display = st.session_state.get(f"{BASE_FILE_SELECT_KEY}_display", CUSTOM_LABEL)
    normalized = _selector_value_to_name(selected_display)
    if st.session_state.get(BASE_FILE_SELECT_KEY) != normalized:
        st.session_state[BASE_FILE_SELECT_KEY] = normalized
        _on_base_file_change()


def _sync_base_editor(base_path: Path) -> None:
    """Load base config text into session state when source path changes."""
    prev = st.session_state.get(BASE_PATH_KEY)
    if prev == str(base_path) and BASE_EDITOR_KEY in st.session_state:
        return
    st.session_state[BASE_PATH_KEY] = str(base_path)
    if base_path.exists():
        st.session_state[BASE_EDITOR_KEY] = base_path.read_text(encoding="utf-8")
    else:
        st.session_state[BASE_EDITOR_KEY] = ""


def _validate_base_yaml(text: str) -> tuple[str | None, dict | None]:
    """Validate editable base config text for the sweep."""
    if not text.strip():
        return "Base config editor is empty.", None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return f"Base config YAML parse error: {e}", None
    if not isinstance(data, dict):
        return "Base config top-level YAML must be a mapping.", None
    # Sweep expansion assigns fresh run IDs. Dropping any base run_id also keeps
    # the UI compatible with older imported config schemas during hot reloads.
    data.pop("run_id", None)
    if isinstance(data.get("environment"), dict):
        data["environment"].pop("_calibration_note", None)
    try:
        cfg = ExperimentConfig(**data)
        return (
            f"Base config valid — env `{cfg.env_type}`, "
            f"{cfg.environment.n_agents} agents, "
            f"{cfg.environment.n_rounds} rounds.",
            data,
        )
    except ValidationError as e:
        return f"Base config validation error:\n{e}", None
    except Exception as e:
        return f"Base config {type(e).__name__}: {e}", None


def _patch_top_level(text: str, key: str, value: str) -> str:
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return text
    if not isinstance(data, dict):
        return text
    data[key] = value
    return yaml.safe_dump(data, sort_keys=False)


def _patch_oversight_mode(text: str, mode: str) -> str:
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return text
    if not isinstance(data, dict):
        return text
    oversight = data.get("oversight")
    if not isinstance(oversight, dict):
        oversight = {}
    oversight["mode"] = mode
    data["oversight"] = oversight
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


def _patch_agent_models(text: str, updates: dict[str, str]) -> str:
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return text
    if not isinstance(data, dict):
        return text
    agents = data.get("agents")
    if isinstance(agents, dict):
        agents.update(updates)
        data["agents"] = agents
    elif isinstance(agents, list):
        for entry in agents:
            if isinstance(entry, dict):
                entry.update(updates)
        data["agents"] = agents
    else:
        return text
    return yaml.safe_dump(data, sort_keys=False)


def _render_base_model_picker() -> None:
    text = st.session_state.get(BASE_EDITOR_KEY, "")
    _, base_data = _validate_base_yaml(text)
    if not isinstance(base_data, dict):
        return
    cfg = ExperimentConfig(**base_data)
    model_options = list(MODEL_TO_BACKEND.keys())
    if cfg.agents.model not in model_options:
        model_options = [cfg.agents.model] + model_options
    selected_model = st.selectbox(
        "Firm model",
        options=model_options,
        index=model_options.index(cfg.agents.model),
        key="sweep_base_firm_model",
    )
    selected_backend = MODEL_TO_BACKEND.get(selected_model, cfg.agents.backend)
    if selected_model != cfg.agents.model or selected_backend != cfg.agents.backend:
        st.session_state[BASE_EDITOR_KEY] = _patch_agent_models(
            text,
            {"backend": selected_backend, "model": selected_model},
        )
        st.rerun()


def _render_base_auditor_model_picker() -> None:
    text = st.session_state.get(BASE_EDITOR_KEY, "")
    _, base_data = _validate_base_yaml(text)
    if not isinstance(base_data, dict):
        return
    cfg = ExperimentConfig(**base_data)
    oversight = cfg.oversight
    if oversight.mode != "audit-penalty" or not oversight.llm_judge_enabled:
        return
    judge_model_options = list(MODEL_TO_BACKEND.keys())
    if oversight.llm_judge_model not in judge_model_options:
        judge_model_options = [oversight.llm_judge_model] + judge_model_options
    selected_judge_model = st.selectbox(
        "Auditor model",
        options=judge_model_options,
        index=judge_model_options.index(oversight.llm_judge_model),
        key="sweep_base_auditor_model",
    )
    selected_judge_backend = MODEL_TO_BACKEND.get(selected_judge_model, oversight.llm_judge_backend)
    if (
        selected_judge_model != oversight.llm_judge_model
        or selected_judge_backend != oversight.llm_judge_backend
    ):
        st.session_state[BASE_EDITOR_KEY] = _patch_auditor_models(
            text,
            {
                "llm_judge_backend": selected_judge_backend,
                "llm_judge_model": selected_judge_model,
            },
        )
        st.rerun()


def _patch_environment(text: str, updates: dict) -> str:
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return text
    if not isinstance(data, dict):
        return text
    env = data.get("environment")
    if not isinstance(env, dict):
        return text
    env.update(updates)
    data["environment"] = env
    return yaml.safe_dump(data, sort_keys=False)


def _render_base_mode_picker() -> None:
    text = st.session_state.get(BASE_EDITOR_KEY, "")
    _, base_data = _validate_base_yaml(text)
    if not isinstance(base_data, dict):
        return
    cfg = ExperimentConfig(**base_data)
    c1, c2 = st.columns(2)
    selected_comm = c1.selectbox(
        "Communication mode",
        options=_COMM_OPTIONS,
        index=_COMM_OPTIONS.index(cfg.communication_mode),
        key="sweep_base_comm_mode",
    )
    selected_audit = c2.selectbox(
        "Audit mode",
        options=_AUDIT_OPTIONS,
        index=_AUDIT_OPTIONS.index(cfg.oversight.mode),
        key="sweep_base_audit_mode",
    )
    changed = False
    editor = text
    if selected_comm != cfg.communication_mode:
        editor = _patch_top_level(editor, "communication_mode", selected_comm)
        changed = True
    if selected_audit != cfg.oversight.mode:
        editor = _patch_oversight_mode(editor, selected_audit)
        changed = True
    if cfg.oversight.mode == "audit-penalty":
        audit_prob = st.slider(
            "Audit probability",
            min_value=0.0,
            max_value=1.0,
            value=float(cfg.oversight.audit_probability),
            step=0.05,
            key="sweep_base_audit_probability",
        )
        if audit_prob != cfg.oversight.audit_probability:
            editor = _patch_auditor_models(editor, {"audit_probability": audit_prob})
            changed = True
    if changed:
        st.session_state[BASE_EDITOR_KEY] = editor
        st.rerun()


def _render_base_game_controls() -> None:
    text = st.session_state.get(BASE_EDITOR_KEY, "")
    _, base_data = _validate_base_yaml(text)
    if not isinstance(base_data, dict):
        return
    cfg = ExperimentConfig(**base_data)
    c1, c2 = st.columns(2)
    selected_rounds = c1.number_input(
        "Rounds",
        min_value=1,
        max_value=1000,
        value=cfg.environment.n_rounds,
        step=1,
        key="sweep_base_n_rounds",
    )
    selected_agents = c2.number_input(
        "Agents",
        min_value=2,
        max_value=10,
        value=cfg.environment.n_agents,
        step=1,
        key="sweep_base_n_agents",
    )
    changed = False
    editor = text
    if int(selected_rounds) != cfg.environment.n_rounds:
        editor = _patch_environment(editor, {"n_rounds": int(selected_rounds)})
        changed = True
    if int(selected_agents) != cfg.environment.n_agents:
        editor = _patch_environment(editor, {"n_agents": int(selected_agents)})
        changed = True
    if changed:
        st.session_state[BASE_EDITOR_KEY] = editor
        st.rerun()


def _materialize_ui_base_config(text: str) -> Path:
    """Write edited base config to a stable temp file for this UI session."""
    tmp_dir = Path("data/raw/_ui_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        data = yaml.safe_load(text) or {}
        if isinstance(data, dict):
            data.pop("run_id", None)
            text = yaml.safe_dump(data, sort_keys=False)
    except yaml.YAMLError:
        pass
    digest = sha1(text.encode("utf-8")).hexdigest()[:16]
    path = tmp_dir / f"base_{digest}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


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
    statuses = []
    for i in range(n_total):
        status = "succeeded/failed" if i < n_done else "queued/running"
        statuses.append({"run": i + 1, "status": status})
    st.subheader("Per-run Status")
    st.dataframe(pd.DataFrame(statuses), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------


def _render_completed(state: dict[str, Any]) -> None:
    def _run_preview(entry: dict[str, Any]) -> str:
        run_id = str(entry.get("run_id") or "unknown")
        run_id_preview = f"{run_id[:8]}...{run_id[-4:]}" if len(run_id) > 12 else run_id
        cfg = entry.get("config") or {}
        env = cfg.get("environment", {}) if isinstance(cfg, dict) else {}
        oversight = cfg.get("oversight", {}) if isinstance(cfg, dict) else {}
        agents_raw = cfg.get("agents", {}) if isinstance(cfg, dict) else {}
        if isinstance(agents_raw, dict):
            agents = agents_raw
        elif isinstance(agents_raw, list) and agents_raw and isinstance(agents_raw[0], dict):
            agents = agents_raw[0]
        else:
            agents = {}

        chunks = [
            f"run_id={run_id_preview}",
            f"model={agents.get('model', 'unknown')}",
            f"rounds={env.get('n_rounds', '?')}",
            f"agents={env.get('n_agents', '?')}",
            f"comm={cfg.get('communication_mode', 'unknown')}",
            f"oversight={oversight.get('mode', 'unknown')}",
        ]
        if oversight.get("mode") == "audit-penalty" and oversight.get("audit_probability") is not None:
            chunks.append(f"p_audit={float(oversight['audit_probability']):.2f}")
        return " | ".join(chunks)

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
            rows.append({
                "run": _run_preview(r),
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
        if state.get("manifest_path"):
            st.session_state["_selected_sweep_path"] = state["manifest_path"]
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

    base_path = Path(sweep_cfg.base_config)
    _render_base_selector(base_path)
    selected_base = st.session_state.get(BASE_FILE_SELECT_KEY, CUSTOM_LABEL)
    if selected_base != CUSTOM_LABEL:
        _sync_base_editor(CONFIGS_DIR / selected_base)
    else:
        _sync_base_editor(base_path)
    st.subheader("Base Experiment Config")
    st.caption("Edits here are applied to preview + launch for this sweep run.")
    # Firm model picker must render *before* the Base Config YAML text area so
    # patching can assign `BASE_EDITOR_KEY` without violating Streamlit's rule
    # against mutating a widget key after that widget instantiates this run.
    _render_base_model_picker()
    _render_base_mode_picker()
    _render_base_game_controls()
    _render_base_auditor_model_picker()
    st.text_area(
        "Base Config YAML",
        height=260,
        key=BASE_EDITOR_KEY,
    )

    base_msg, base_data = _validate_base_yaml(st.session_state.get(BASE_EDITOR_KEY, ""))
    if base_data is None:
        st.warning(base_msg)
        return
    st.success(base_msg)

    effective_base_path = _materialize_ui_base_config(
        st.session_state.get(BASE_EDITOR_KEY, "")
    )
    sweep_data = sweep_cfg.model_dump(mode="python")
    sweep_data["base_config"] = str(effective_base_path)
    effective_sweep_cfg = SweepConfig(**sweep_data)

    configs, expand_err = _try_expand(effective_sweep_cfg)
    if expand_err:
        st.warning(expand_err)
        return

    _render_grid_preview(effective_sweep_cfg, configs)

    # Launch controls
    max_workers = st.number_input(
        "Max workers",
        min_value=1,
        max_value=32,
        value=min(os.cpu_count() or 1, 4),
        step=1,
        key=WORKERS_KEY,
        help="Number of parallel worker processes for the sweep.",
    )

    if st.button("Launch sweep", type="primary"):
        _start_sweep(effective_sweep_cfg, int(max_workers), len(configs))
        st.rerun()
