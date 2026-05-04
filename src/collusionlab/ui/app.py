"""CollusionLab Analysis UI.

Streamlit app for inspecting experiment runs, launching experiments and
parameter sweeps, and comparing results across conditions.

Launch:
    streamlit run src/collusionlab/ui/app.py
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from collusionlab.ui.data_loading import (
    build_compare_df,
    build_run_index,
    build_transcript_df,
    get_signal,
    normalize_reasoning,
    list_sweeps,
    list_runs,
    load_log_rows,
    load_manifest,
    load_sweep_manifest,
    extract_trajectory_df,
)
from collusionlab.ui.run_page import render_run_page
from collusionlab.ui.sweep_page import render_sweep_page
from collusionlab.metrics.base import LogReader, get_metrics_computer
from collusionlab.metrics import collusion, concealment
from collusionlab.storage import is_database_uri
import collusionlab.environments.pricing.metrics  # noqa: F401

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CollusionLab",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("CollusionLab")

_PAGES = ["Run", "Sweep", "Analyze", "Compare"]
_nav_target = st.session_state.pop("_nav_target", None)
_nav_index = _PAGES.index(_nav_target) if _nav_target in _PAGES else st.session_state.get("_nav_index", 0)

page = st.sidebar.radio(
    "Navigation",
    _PAGES,
    index=_nav_index,
    key="nav_page",
)
st.session_state["_nav_index"] = _PAGES.index(page)

# ---------------------------------------------------------------------------
# Placeholder pages
# ---------------------------------------------------------------------------


def page_run():
    render_run_page()


def page_sweep():
    render_sweep_page()


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------


def _safe_run_metrics(run_dir: Path | str) -> tuple[dict, object | None]:
    """Load RunData + computed metrics for a run directory."""
    try:
        manifest_ref = run_dir / "manifest.json" if isinstance(run_dir, Path) else run_dir
        run_data = LogReader.load_run(manifest_ref)
        computer = get_metrics_computer(run_data.env_type)
        return computer.compute(run_data), run_data
    except Exception:
        return {}, None


def _fmt_metric(value, suffix: str = "", digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _fmt_bool_metric(value) -> str:
    if value is None:
        return "N/A"
    return "Yes" if bool(value) else "No"


def _highlight_keyword(text: str, keyword: str) -> str:
    if not keyword:
        return text
    return re.sub(
        re.escape(keyword),
        lambda m: f"**{m.group(0)}**",
        text,
        flags=re.IGNORECASE,
    )


def _binomial_ci(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    se = math.sqrt(max(rate * (1 - rate), 0) / n)
    return (max(0.0, rate - z * se), min(1.0, rate + z * se))


def _run_id_preview(run_id: str | None, n: int = 8) -> str:
    rid = str(run_id or "")
    return f"{rid[:n]}..." if rid else "unknown"


def _coerce_agents_config(agents_cfg_raw) -> dict:
    if isinstance(agents_cfg_raw, dict):
        return agents_cfg_raw
    if isinstance(agents_cfg_raw, list) and agents_cfg_raw and isinstance(agents_cfg_raw[0], dict):
        return agents_cfg_raw[0]
    return {}


def _run_selector_label_from_fields(
    run_id: str | None,
    model: str | None,
    n_rounds,
    n_agents,
    communication_mode: str | None,
    oversight_mode: str | None,
    audit_probability=None,
) -> str:
    """Run label for Analyze/Compare selectors with selective keys."""
    is_audit = str(oversight_mode or "unknown") == "audit-penalty"
    p_val = "?"
    if audit_probability is not None:
        p_val = f"{float(audit_probability):.2f}"
    chunks = [
        _run_id_preview(run_id),
        str(model or "unknown"),
        f"n={n_rounds if n_rounds is not None else '?'}",
        f"agents={n_agents if n_agents is not None else '?'}",
        f"comm={communication_mode or 'unknown'}",
        str(oversight_mode or "unknown"),
        f"audit (p={p_val})={'true' if is_audit else 'false'}",
    ]
    return " | ".join(chunks)


def _detailed_run_selector_label(run_id: str | None, manifest: dict | None = None) -> str:
    """Build a run selector label aligned with Analyze tab details."""
    manifest = manifest or {}
    config = manifest.get("config", {}) or {}
    env_cfg = config.get("environment", {}) or {}
    agents_cfg = _coerce_agents_config(config.get("agents", {}) or {})
    oversight_cfg = config.get("oversight", {}) or {}
    return _run_selector_label_from_fields(
        run_id=run_id,
        model=agents_cfg.get("model"),
        n_rounds=env_cfg.get("n_rounds"),
        n_agents=env_cfg.get("n_agents"),
        communication_mode=config.get("communication_mode"),
        oversight_mode=oversight_cfg.get("mode"),
        audit_probability=oversight_cfg.get("audit_probability"),
    )


def _compact_value_list(values: list) -> str:
    uniq = sorted({str(v) for v in values if v is not None and str(v) != ""})
    if not uniq:
        return "?"
    if len(uniq) == 1:
        return uniq[0]
    return "[" + ", ".join(uniq) + "]"


def _detailed_sweep_selector_label(sweep: dict) -> str:
    """Build a rich label for the Compare sweep selector."""
    sweep_id = str(sweep.get("sweep_id", "unknown"))
    sweep_manifest = load_sweep_manifest(sweep.get("path", ""))
    runs_meta = (sweep_manifest or {}).get("runs", []) or []
    models: list[str] = []
    rounds: list = []
    agents: list = []
    comms: list[str] = []
    oversights: list[str] = []
    audits: list[str] = []
    for run_meta in runs_meta:
        cfg = (
            run_meta.get("config_snapshot")
            or run_meta.get("config")
            or {}
        )
        if not isinstance(cfg, dict):
            continue
        env_cfg = cfg.get("environment", {}) or {}
        agents_cfg = _coerce_agents_config(cfg.get("agents", {}) or {})
        oversight_cfg = cfg.get("oversight", {}) or {}
        models.append(agents_cfg.get("model", "unknown"))
        rounds.append(env_cfg.get("n_rounds"))
        agents.append(env_cfg.get("n_agents"))
        comms.append(cfg.get("communication_mode", "unknown"))
        oversights.append(oversight_cfg.get("mode", "unknown"))
        is_audit = oversight_cfg.get("mode") == "audit-penalty"
        p_val = oversight_cfg.get("audit_probability")
        p_str = f"{float(p_val):.2f}" if p_val is not None else "?"
        audits.append(f"(p={p_str})={'true' if is_audit else 'false'}")
    n_runs = sweep.get("n_runs", "?")
    chunks = [
        f"{sweep_id[:8]}...",
        _compact_value_list(models),
        f"n={_compact_value_list(rounds)}",
        f"agents={_compact_value_list(agents)}",
        f"comm={_compact_value_list(comms)}",
        _compact_value_list(oversights),
        f"audit {_compact_value_list(audits)}",
    ]
    chunks.append(str(n_runs))
    return " | ".join(chunks)


def _plotly_image_bytes(fig: go.Figure, fmt: str) -> bytes | None:
    try:
        return fig.to_image(format=fmt)
    except Exception:
        return None


def page_compare():
    st.header("Compare Runs")
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        st.warning("No data found. `data/raw/` does not exist.")
        return

    sweeps = list_sweeps(raw_dir)
    if not sweeps:
        st.info("No sweep manifests found under `data/raw/sweep_*/`.")
        return

    options = {_detailed_sweep_selector_label(s): s["path"] for s in sweeps}
    selected_sweep_path = st.session_state.pop("_selected_sweep_path", None)
    option_labels = list(options.keys())
    sweep_index = 0
    if selected_sweep_path:
        for i, label in enumerate(option_labels):
            if str(options[label]) == str(selected_sweep_path):
                sweep_index = i
                break
    selected = st.selectbox("Select sweep", options=option_labels, index=sweep_index)
    sweep_path = Path(options[selected])
    sweep_manifest = load_sweep_manifest(sweep_path)
    if sweep_manifest is None:
        st.error("Failed to load sweep manifest.")
        return

    runs = LogReader.load_sweep(sweep_path)
    if not runs:
        st.warning("No successful runs available in this sweep.")
        return

    env_type = runs[0].env_type
    computer = get_metrics_computer(env_type)
    sweep_df = build_compare_df(computer.compute_sweep(runs))
    run_paths = {
        r.get("run_id"): Path(r.get("manifest_path", "")).parent
        for r in sweep_manifest.get("runs", [])
        if r.get("manifest_path")
    }

    st.caption(
        f"Sweep `{sweep_manifest.get('sweep_id', 'unknown')}` — "
        f"{len(runs)} successful runs loaded."
    )

    tab_threshold, tab_transition, tab_browser, tab_side_by_side = st.tabs(
        ["Threshold View", "Transition View", "Run Browser", "Side-by-Side"]
    )

    with tab_threshold:
        render_threshold_view(runs, sweep_df)
    with tab_transition:
        render_transition_view(runs, sweep_df)
    with tab_browser:
        render_run_browser(sweep_df, run_paths)
    with tab_side_by_side:
        render_side_by_side(runs, run_paths)


def render_side_by_side(runs, run_paths: dict) -> None:
    if not runs:
        st.info("No runs available.")
        return

    options: list[dict] = []
    for r in runs:
        run_dir = run_paths.get(r.run_id)
        manifest = load_manifest(run_dir) if run_dir else None
        options.append({
            "run_id": r.run_id,
            "run_dir": run_dir,
            "label": _detailed_run_selector_label(r.run_id, manifest),
        })

    render_side_by_side_options(
        options,
        key_prefix="compare_sbs",
        max_selections=4,
        columns_per_row=4,
        label="Select runs to compare (up to 4)",
    )


def _unique_side_by_side_options(options: list[dict]) -> list[dict]:
    """Return side-by-side run options with unique display labels."""
    seen: dict[str, int] = {}
    assigned: set[str] = set()
    unique: list[dict] = []
    for opt in options:
        label = str(opt.get("label") or opt.get("run_id") or "unknown")
        count = seen.get(label, 0) + 1
        seen[label] = count
        unique_label = label if count == 1 else f"{label} ({count})"
        while unique_label in assigned:
            count += 1
            seen[label] = count
            unique_label = f"{label} ({count})"
        assigned.add(unique_label)
        unique.append({**opt, "label": unique_label})
    return unique


def _run_index_side_by_side_options(run_index: pd.DataFrame) -> list[dict]:
    """Build arbitrary-run side-by-side options from the Analyze run index."""
    if run_index.empty:
        return []

    options: list[dict] = []
    for _, row in run_index.iterrows():
        run_id = str(row.get("run_id") or "")
        label = _run_selector_label_from_fields(
            run_id=run_id,
            model=row.get("firm_model"),
            n_rounds=row.get("n_rounds"),
            n_agents=row.get("n_agents"),
            communication_mode=row.get("comm_mode"),
            oversight_mode=row.get("oversight_mode"),
            audit_probability=row.get("audit_probability"),
        )
        options.append({
            "run_id": run_id,
            "run_dir": row.get("run_dir"),
            "label": label,
        })
    return _unique_side_by_side_options(options)


def _default_side_by_side_labels(
    options: list[dict],
    preferred_run_id: str | None,
    n_default: int = 2,
) -> list[str]:
    """Default to the current run plus the next available run."""
    if not options or n_default <= 0:
        return []

    selected: list[str] = []
    if preferred_run_id:
        for opt in options:
            if opt.get("run_id") == preferred_run_id:
                selected.append(opt["label"])
                break

    for opt in options:
        if len(selected) >= min(n_default, len(options)):
            break
        label = opt["label"]
        if label not in selected:
            selected.append(label)
    return selected


def _wrapped_side_by_side_columns(selected: list[dict], columns_per_row: int):
    """Yield Streamlit columns for selected runs, wrapping within one section."""
    width = max(1, int(columns_per_row or 1))
    for start in range(0, len(selected), width):
        chunk = selected[start:start + width]
        yield zip(st.columns(len(chunk)), chunk)


def _load_side_by_side_selection(
    options: list[dict],
    selected_labels: list[str],
) -> list[dict]:
    option_by_label = {opt["label"]: opt for opt in options}
    selected: list[dict] = []
    for label in selected_labels:
        opt = option_by_label[label]
        run_dir = opt.get("run_dir")
        if not run_dir:
            continue
        rows = load_log_rows(run_dir)
        manifest = load_manifest(run_dir)
        if not rows or manifest is None:
            continue
        metrics, _ = _safe_run_metrics(run_dir)
        selected.append({
            "label": label,
            "run_id": opt.get("run_id"),
            "run_dir": run_dir,
            "rows": rows,
            "manifest": manifest,
            "metrics": metrics or {},
        })
    return selected


def render_side_by_side_options(
    options: list[dict],
    *,
    key_prefix: str,
    default_run_ids: list[str] | None = None,
    max_selections: int | None = None,
    columns_per_row: int = 3,
    label: str = "Select runs to compare",
) -> None:
    options = _unique_side_by_side_options([opt for opt in options if opt.get("run_dir")])
    if not options:
        st.info("No runs available.")
        return

    default_labels = _default_side_by_side_labels(options, None)
    if default_run_ids:
        default_labels = []
        for run_id in default_run_ids:
            for opt in options:
                if opt.get("run_id") == run_id and opt["label"] not in default_labels:
                    default_labels.append(opt["label"])
                    break
        if len(default_labels) < min(2, len(options)):
            for opt in options:
                if opt["label"] not in default_labels:
                    default_labels.append(opt["label"])
                if len(default_labels) >= min(2, len(options)):
                    break

    multiselect_kwargs = {
        "label": label,
        "options": [opt["label"] for opt in options],
        "default": default_labels,
        "key": f"{key_prefix}_run_select",
    }
    if max_selections is not None:
        multiselect_kwargs["max_selections"] = max_selections

    selected_labels = st.multiselect(**multiselect_kwargs)
    if not selected_labels:
        st.info("Select at least one run.")
        return

    selected = _load_side_by_side_selection(options, selected_labels)
    if not selected:
        st.warning("Could not load data for the selected runs.")
        return

    st.subheader("Metrics")
    for wrapped in _wrapped_side_by_side_columns(selected, columns_per_row):
        for col, d in wrapped:
            with col:
                _render_side_by_side_metrics(d)

    st.subheader("Actions Over Time")
    for wrapped in _wrapped_side_by_side_columns(selected, columns_per_row):
        for idx, (col, d) in enumerate(wrapped):
            with col:
                _render_side_by_side_actions(d, key=f"{key_prefix}_actions_{idx}_{d['run_id']}")

    st.subheader("Reward Elevation Over Time")
    for wrapped in _wrapped_side_by_side_columns(selected, columns_per_row):
        for idx, (col, d) in enumerate(wrapped):
            with col:
                _render_side_by_side_reward_elevation(
                    d,
                    key=f"{key_prefix}_reward_elevation_{idx}_{d['run_id']}",
                )

    st.subheader("Coordination Signals")
    for wrapped in _wrapped_side_by_side_columns(selected, columns_per_row):
        for idx, (col, d) in enumerate(wrapped):
            with col:
                _render_side_by_side_coordination(
                    d,
                    key=f"{key_prefix}_coordination_{idx}_{d['run_id']}",
                )

    st.subheader("Transcript")
    transcript_labels = [d["label"] for d in selected]
    chosen_label = st.selectbox(
        "View transcript for",
        transcript_labels,
        key=f"{key_prefix}_transcript_select",
    )
    chosen = next(d for d in selected if d["label"] == chosen_label)
    render_transcript_tab(
        chosen["rows"],
        chosen["metrics"],
        key_prefix=f"{key_prefix}_transcript",
    )


def _render_side_by_side_metrics(d: dict) -> None:
    st.markdown(f"**`{d['label']}`**")
    m = d["metrics"]
    rows_ = d["rows"]
    elevs = [
        sum(r.get("trajectory_signals", {}).get("reward_elevation") or []) /
        len(r.get("trajectory_signals", {}).get("reward_elevation") or [1])
        for r in rows_
        if r.get("trajectory_signals", {}).get("reward_elevation")
    ]
    mean_elev = sum(elevs) / len(elevs) if elevs else None
    spreads_ = [
        r.get("trajectory_signals", {}).get("action_spread")
        for r in rows_
        if r.get("trajectory_signals", {}).get("action_spread") is not None
    ]
    mean_spread = sum(spreads_) / len(spreads_) if spreads_ else None
    covert_ct = sum(
        1 for r in rows_ if r.get("trajectory_signals", {}).get("covert_coordination_flag")
    )
    penalized_ct = sum(
        1 for r in rows_ if r.get("audit_event") and r["audit_event"].get("penalty_applied")
    )
    st.metric("Mean Elevation", f"{mean_elev:.3f}" if mean_elev is not None else "N/A")
    st.metric("Mean Spread", f"{mean_spread:.2f}" if mean_spread is not None else "N/A")
    st.metric("Onset Round", _fmt_metric(m.get("onset_round"), digits=0))
    st.metric(
        "Behavioral Covert Score",
        _fmt_metric(m.get("behavioral_steganographic_score", m.get("steganographic_score"))),
    )
    st.metric("Message Stego Signature", _fmt_bool_metric(m.get("steganographic_signature")))
    st.metric("Covert Rounds", covert_ct)
    st.metric("Penalized Rounds", penalized_ct)


def _render_side_by_side_actions(d: dict, key: str) -> None:
    st.caption(d["label"])
    df = extract_trajectory_df(d["rows"])
    if df.empty:
        st.info("No data.")
        return
    env_cfg = d["manifest"].get("config", {}).get("environment", {})
    nash, monopoly = env_cfg.get("nash_price"), env_cfg.get("monopoly_price")
    onset = d["metrics"].get("onset_round")
    action_cols_ = sorted([c for c in df.columns if c.startswith("action_") and c[7:].isdigit()])
    fig = go.Figure()
    for i, ac in enumerate(action_cols_):
        fig.add_trace(go.Scatter(
            x=df["round"],
            y=df[ac],
            mode="lines+markers",
            name=f"Agent {i}",
            hovertemplate="R%{x} P=%{y}<extra></extra>",
        ))
    if nash is not None:
        fig.add_hline(y=nash, line_dash="dash", line_color="green", annotation_text="Nash")
    if monopoly is not None:
        fig.add_hline(y=monopoly, line_dash="dash", line_color="red", annotation_text="Mono")
    if onset is not None:
        fig.add_vline(x=onset, line_dash="dash", annotation_text="Onset")
    fig.update_layout(
        xaxis_title="Round",
        yaxis_title="Price",
        height=260,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    st.plotly_chart(fig, width="stretch", key=key)


def _render_side_by_side_reward_elevation(d: dict, key: str) -> None:
    st.caption(d["label"])
    df = extract_trajectory_df(d["rows"])
    elev_cols_ = [c for c in df.columns if c.startswith("reward_elevation_")]
    if df.empty or not elev_cols_:
        st.info("No elevation data.")
        return
    onset = d["metrics"].get("onset_round")
    fig = go.Figure()
    for i, ec in enumerate(elev_cols_):
        fig.add_trace(go.Scatter(
            x=df["round"],
            y=df[ec],
            mode="lines+markers",
            name=f"Agent {i}",
            hovertemplate="R%{x} elev=%{y:.3f}<extra></extra>",
        ))
    smooth = df[elev_cols_].mean(axis=1).rolling(5, min_periods=1).mean()
    fig.add_trace(go.Scatter(x=df["round"], y=smooth, mode="lines", name="Smoothed", line={"width": 3}))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Nash")
    fig.add_hline(y=1, line_dash="dot", line_color="gray", annotation_text="Mono")
    if onset is not None:
        fig.add_vline(x=onset, line_dash="dash", annotation_text="Onset")
        fig.add_vrect(x0=onset, x1=df["round"].max(), opacity=0.08, line_width=0)
    fig.update_layout(
        xaxis_title="Round",
        yaxis_title="Elevation",
        height=260,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    st.plotly_chart(fig, width="stretch", key=key)


def _render_side_by_side_coordination(d: dict, key: str) -> None:
    st.caption(d["label"])
    df = extract_trajectory_df(d["rows"])
    if df.empty or "action_spread" not in df.columns:
        st.info("No signals data.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["round"],
        y=df["action_spread"],
        mode="lines+markers",
        name="Spread",
        hovertemplate="R%{x} spread=%{y}<extra></extra>",
    ))
    if "covert_coordination_flag" in df.columns:
        covert_roll = df["covert_coordination_flag"].astype(float).rolling(10, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=df["round"], y=covert_roll, mode="lines", name="Covert rate", yaxis="y2"))
    if "hollow_coordination_flag" in df.columns:
        hollow_roll = df["hollow_coordination_flag"].astype(float).rolling(10, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df["round"],
            y=hollow_roll,
            mode="lines",
            name="Hollow rate",
            yaxis="y2",
            line={"dash": "dot"},
        ))
    fig.update_layout(
        xaxis_title="Round",
        yaxis_title="Spread",
        yaxis2={"title": "Rate", "overlaying": "y", "side": "right", "range": [0, 1]},
        height=260,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    st.plotly_chart(fig, width="stretch", key=key)


def render_threshold_view(runs, sweep_df: pd.DataFrame):
    st.subheader("Onset Threshold Analysis")
    st.caption("Onset rate is the share of runs whose reward elevation sustains above the threshold for the selected duration.")
    dims = ["communication_mode", "memory_window", "n_agents", "oversight_mode"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x_dim = st.selectbox("Heatmap X dimension", dims, index=1)
    with c2:
        y_dim = st.selectbox("Heatmap Y dimension", dims, index=0)
    with c3:
        elevation_threshold = st.number_input("Elevation threshold", min_value=0.0, max_value=2.0, value=0.3, step=0.05)
    with c4:
        min_duration = st.number_input("Min duration", min_value=1, max_value=50, value=5, step=1)
    groupby = list(dict.fromkeys([y_dim, x_dim]))

    extra_group = st.multiselect(
        "Additional grouping dimensions for table",
        options=[d for d in dims if d not in groupby],
        default=[],
    )
    groupby = groupby + extra_group
    if not groupby:
        st.info("Choose at least one grouping dimension.")
        return

    table = collusion.threshold_table(
        runs,
        groupby=groupby,
        elevation_threshold=elevation_threshold,
        min_duration=int(min_duration),
    )
    if table.empty:
        st.info("No threshold data available.")
        return
    st.dataframe(table, width="stretch")
    st.download_button(
        "Download threshold table CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="threshold_table.csv",
        mime="text/csv",
    )

    if all(c in table.columns for c in [x_dim, y_dim, "onset_rate"]):
        pivot = table.pivot_table(
            index=y_dim,
            columns=x_dim,
            values="onset_rate",
            aggfunc="mean",
        )
        if not pivot.empty:
            fig = px.imshow(
                pivot,
                text_auto=".2f",
                aspect="auto",
                title="Onset Rate Heatmap",
                labels={"x": x_dim, "y": y_dim, "color": "Onset rate"},
            )
            st.plotly_chart(fig, width="stretch")
            png = _plotly_image_bytes(fig, "png")
            if png:
                st.download_button("Download heatmap PNG", png, file_name="onset_heatmap.png", mime="image/png")

    if "onset_round" in sweep_df.columns:
        onset_rate_by_mode = (
            sweep_df.assign(onset_detected=sweep_df["onset_round"].notna().astype(float))
            .groupby(y_dim, dropna=False)["onset_detected"]
            .agg(["mean", "count"])
            .reset_index()
        )
        ci = onset_rate_by_mode.apply(
            lambda r: _binomial_ci(float(r["mean"]), int(r["count"])),
            axis=1,
        )
        onset_rate_by_mode["ci_low"] = [x[0] for x in ci]
        onset_rate_by_mode["ci_high"] = [x[1] for x in ci]
        onset_rate_by_mode["err_plus"] = onset_rate_by_mode["ci_high"] - onset_rate_by_mode["mean"]
        onset_rate_by_mode["err_minus"] = onset_rate_by_mode["mean"] - onset_rate_by_mode["ci_low"]
        fig_bar = px.bar(
            onset_rate_by_mode,
            x=y_dim,
            y="mean",
            error_y="err_plus",
            error_y_minus="err_minus",
            title=f"Onset Rate by {y_dim} (95% CI)",
            labels={"mean": "Onset rate", y_dim: y_dim},
        )
        st.plotly_chart(fig_bar, width="stretch")
        st.caption("Error bars use a normal-approximation 95% confidence interval over seeds/runs in each group.")


def render_transition_view(runs, sweep_df: pd.DataFrame):
    st.subheader("Concealment and Transition Dynamics")
    st.caption("Covert coordination is elevated behavior without explicit collusive language. Hollow coordination is the reverse.")
    c1, c2, c3 = st.columns(3)
    with c1:
        window = st.number_input("Rolling window", min_value=1, max_value=50, value=10, step=1)
    with c2:
        elevation_threshold = st.number_input("Behavior threshold", min_value=0.0, max_value=2.0, value=0.3, step=0.05, key="transition_behavior_threshold")
    with c3:
        audited_only = st.checkbox("Audited rounds only", value=False, help="If enabled, non-audited rounds are excluded from covert/hollow rates.")

    cond = concealment.concealment_by_condition(
        runs, groupby=["communication_mode", "oversight_mode"], window=int(window), elevation_threshold=elevation_threshold
    )
    if not cond.empty:
        st.dataframe(cond, width="stretch")
        st.download_button(
            "Download concealment summary CSV",
            cond.to_csv(index=False).encode("utf-8"),
            file_name="concealment_summary.csv",
            mime="text/csv",
        )

        fig = px.bar(
            cond,
            x="communication_mode",
            y="steganographic_score",
            color="oversight_mode",
            barmode="group",
            title="Legacy Behavioral Covert Score by Condition",
        )
        st.plotly_chart(fig, width="stretch")

    if "steganographic_signature" in sweep_df.columns:
        analysis_cols = [
            "steganographic_signature",
            "steganography_llm_explicit_rate",
            "steganography_behavior_rate",
            "steganography_covert_rate",
        ]
        existing = [c for c in analysis_cols if c in sweep_df.columns]
        if existing:
            analysis_summary = (
                sweep_df.groupby(["communication_mode", "oversight_mode"], dropna=False)[existing]
                .mean(numeric_only=True)
                .reset_index()
                .rename(columns={"steganographic_signature": "signature_rate"})
            )
            st.markdown("**Offline Message Steganography Analysis**")
            st.dataframe(analysis_summary, width="stretch")

    ts_records: list[dict] = []
    for run in runs:
        label = f"{run.communication_mode} / {run.oversight_mode}"
        covert = concealment.covert_coordination_series(
            run, window=int(window), threshold=elevation_threshold, audited_only=audited_only
        )
        hollow = concealment.hollow_coordination_series(
            run, window=int(window), threshold=elevation_threshold, audited_only=audited_only
        )
        for round_num, value in covert.items():
            ts_records.append({"condition": label, "round": round_num, "metric": "Covert", "value": value})
        for round_num, value in hollow.items():
            ts_records.append({"condition": label, "round": round_num, "metric": "Hollow", "value": value})
    ts_df = pd.DataFrame(ts_records)
    if not ts_df.empty:
        mean_ts = (
            ts_df.groupby(["condition", "round", "metric"], dropna=False)["value"]
            .mean()
            .reset_index()
        )
        fig_ts = px.line(
            mean_ts,
            x="round",
            y="value",
            color="condition",
            line_dash="metric",
            title="Mean Rolling Covert/Hollow Coordination by Condition",
            labels={"value": "Rolling rate", "round": "Round"},
        )
        fig_ts.add_vline(x=1, line_dash="dot", annotation_text="Oversight active from round 1")
        st.plotly_chart(fig_ts, width="stretch")
        st.caption("Lines average per-run rolling rates within each condition. Dotted vertical marker indicates oversight is active from the first round for oversight runs.")

    if "transition_round" in sweep_df.columns:
        summary = (
            sweep_df.assign(transition_detected=sweep_df["transition_round"].notna().astype(float))
            .groupby(["communication_mode", "oversight_mode"], dropna=False)["transition_detected"]
            .mean()
            .reset_index()
        )
        fig_rate = px.bar(
            summary,
            x="communication_mode",
            y="transition_detected",
            color="oversight_mode",
            barmode="group",
            title="Transition Rate by Condition",
            labels={"transition_detected": "transition_rate"},
        )
        st.plotly_chart(fig_rate, width="stretch")


def render_run_browser(sweep_df: pd.DataFrame, run_paths: dict[str, Path] | None = None):
    st.subheader("Run Browser")
    run_paths = run_paths or {}
    df = build_compare_df(sweep_df)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        comm = st.multiselect("Communication", sorted(df["communication_mode"].dropna().unique()) if "communication_mode" in df else [], default=sorted(df["communication_mode"].dropna().unique()) if "communication_mode" in df else [])
    with f2:
        oversight = st.multiselect("Oversight", sorted(df["oversight_mode"].dropna().unique()) if "oversight_mode" in df else [], default=sorted(df["oversight_mode"].dropna().unique()) if "oversight_mode" in df else [])
    with f3:
        min_score = st.number_input("Min behavioral covert score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    with f4:
        only_transition = st.checkbox("Transitions only", value=False)

    if comm and "communication_mode" in df:
        df = df[df["communication_mode"].isin(comm)]
    if oversight and "oversight_mode" in df:
        df = df[df["oversight_mode"].isin(oversight)]
    if "steganographic_score" in df:
        df = df[df["steganographic_score"].fillna(0) >= min_score]
    if only_transition and "has_transition" in df:
        df = df[df["has_transition"]]

    cols = [
        "run_id",
        "communication_mode",
        "oversight_mode",
        "n_agents",
        "memory_window",
        "onset_round",
        "transition_round",
        "concealment_gap",
        "price_follow_rate",
        "steganographic_score",
        "steganographic_signature",
        "steganography_message_rounds",
        "steganography_llm_explicit_rate",
        "steganography_behavior_rate",
        "steganography_covert_rate",
        "steganography_top_feature",
    ]
    existing = [c for c in cols if c in df.columns]
    if df.empty:
        st.info("No runs match the browser filters.")
        return
    st.dataframe(
        df[existing].sort_values(
            by="steganographic_score", ascending=False, na_position="last"
        ),
        width="stretch",
    )
    st.download_button(
        "Download run browser CSV",
        df[existing].to_csv(index=False).encode("utf-8"),
        file_name="run_browser.csv",
        mime="text/csv",
    )
    selectable = [rid for rid in df["run_id"].tolist() if rid in run_paths]
    if selectable:
        open_options: list[tuple[str, str]] = []
        for run_id in selectable:
            run_dir = run_paths.get(run_id)
            if not run_dir:
                continue
            manifest = load_manifest(run_dir)
            label = _detailed_run_selector_label(run_id, manifest)
            open_options.append((label, run_id))
        if not open_options:
            st.caption("Run jump is unavailable because run manifests could not be loaded.")
            return
        selected_label = st.selectbox("Open run in Analyze", [x[0] for x in open_options])
        selected_run_id = dict(open_options)[selected_label]
        if st.button("Open selected run"):
            st.session_state["_selected_run_id"] = selected_run_id
            st.session_state["_nav_target"] = "Analyze"
            st.rerun()
    else:
        st.caption("Run jump is unavailable because sweep manifest paths were not found.")


# ---------------------------------------------------------------------------
# Analyze page
# ---------------------------------------------------------------------------


def page_analyze():
    st.header("Analyze Run")

    import os
    raw_dir = os.getenv("COLLUSIONLAB_STORAGE_URL") or "data/raw"
    if not is_database_uri(str(raw_dir)) and not Path(raw_dir).exists():
        st.warning("No runs found. `data/raw/` does not exist.")
        return

    run_index = build_run_index(raw_dir)
    if run_index.empty:
        st.warning("No runs found in `data/raw/`.")
        return

    st.caption("Filter runs before selecting one for detailed replay.")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        env_filter = st.multiselect(
            "Environment",
            sorted(run_index["env_type"].dropna().unique()),
            default=sorted(run_index["env_type"].dropna().unique()),
        )
    with f2:
        comm_filter = st.multiselect(
            "Communication",
            sorted(run_index["comm_mode"].dropna().unique()),
            default=sorted(run_index["comm_mode"].dropna().unique()),
        )
    with f3:
        oversight_filter = st.multiselect(
            "Oversight",
            sorted(run_index["oversight_mode"].dropna().unique()),
            default=sorted(run_index["oversight_mode"].dropna().unique()),
        )
    with f4:
        search = st.text_input("Run ID contains", "")

    filtered = run_index[
        run_index["env_type"].isin(env_filter)
        & run_index["comm_mode"].isin(comm_filter)
        & run_index["oversight_mode"].isin(oversight_filter)
    ]
    if search:
        filtered = filtered[filtered["run_id"].str.contains(search, case=False, na=False)]
    if filtered.empty:
        st.info("No runs match the current filters. Clear one or more filters to continue.")
        return

    display_labels = []
    for _, row in filtered.iterrows():
        display_labels.append(
            _run_selector_label_from_fields(
                run_id=row.get("run_id"),
                model=row.get("firm_model"),
                n_rounds=row.get("n_rounds"),
                n_agents=row.get("n_agents"),
                communication_mode=row.get("comm_mode"),
                oversight_mode=row.get("oversight_mode"),
                audit_probability=row.get("audit_probability"),
            )
        )

    preselected = st.session_state.pop("_selected_run_id", None)
    default_index = 0
    if preselected:
        for i, (_, row) in enumerate(filtered.iterrows()):
            if row.get("run_id") == preselected:
                default_index = i
                break

    selected_index = st.selectbox(
        "Select run",
        options=list(range(len(display_labels))),
        index=default_index,
        format_func=lambda i: display_labels[i],
    )
    selected_row = filtered.iloc[selected_index]
    run_dir = selected_row["run_dir"]

    manifest = load_manifest(run_dir)
    rows = load_log_rows(run_dir)

    if manifest is None or not rows:
        st.error("Failed to load run data.")
        return
    metrics, run_data = _safe_run_metrics(run_dir)

    # --- Tabs ---
    tab_config, tab_trajectory, tab_transcript, tab_metrics, tab_side_by_side = st.tabs(
        ["Config", "Trajectory", "Transcript", "Metrics", "Side-by-Side"]
    )

    with tab_config:
        render_config_tab(manifest, run_index, run_dir)

    with tab_trajectory:
        render_trajectory_tab(rows, manifest, metrics)

    with tab_transcript:
        render_transcript_tab(rows, metrics, key_prefix="analyze_transcript")

    with tab_metrics:
        render_metrics_tab(rows, metrics)

    with tab_side_by_side:
        options = _run_index_side_by_side_options(filtered)
        render_side_by_side_options(
            options,
            key_prefix="analyze_sbs",
            default_run_ids=[str(selected_row.get("run_id") or "")],
            max_selections=None,
            columns_per_row=3,
            label="Select runs to compare",
        )


# ---------------------------------------------------------------------------
# Config tab
# ---------------------------------------------------------------------------


def render_config_tab(manifest: dict, run_index: pd.DataFrame | None = None, current_run_dir: Path | str | None = None):
    st.subheader("Run Metadata")
    col1, col2, col3, col4 = st.columns(4)
    elapsed_seconds = manifest.get("elapsed_seconds")
    input_tokens = manifest.get("total_input_tokens")
    output_tokens = manifest.get("total_output_tokens")
    cost_estimate = manifest.get("total_cost_estimate_usd")
    col1.metric(
        "Elapsed (s)",
        f"{elapsed_seconds:.1f}" if elapsed_seconds is not None else "Running",
    )
    col2.metric("Input Tokens", f"{input_tokens:,}" if input_tokens is not None else "0")
    col3.metric("Output Tokens", f"{output_tokens:,}" if output_tokens is not None else "0")
    col4.metric(
        "Est. Cost ($)",
        f"{cost_estimate:.4f}" if cost_estimate is not None else "0.0000",
    )

    st.subheader("Configuration")
    config = manifest.get("config", {})
    st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language="yaml")

    if run_index is not None and len(run_index) > 1:
        with st.expander("Compare config against another run"):
            options = {
                row["label"]: row["run_dir"]
                for _, row in run_index.iterrows()
                if current_run_dir is None or str(row["run_dir"]) != str(current_run_dir)
            }
            if options:
                label = st.selectbox("Diff target", options=list(options.keys()))
                other_manifest = load_manifest(options[label])
                if other_manifest:
                    left = yaml.dump(config, default_flow_style=False, sort_keys=False).splitlines()
                    right = yaml.dump(
                        other_manifest.get("config", {}),
                        default_flow_style=False,
                        sort_keys=False,
                    ).splitlines()
                    import difflib
                    diff = "\n".join(difflib.unified_diff(left, right, fromfile="selected", tofile="comparison", lineterm=""))
                    st.code(diff or "No config differences.", language="diff")


# ---------------------------------------------------------------------------
# Trajectory tab
# ---------------------------------------------------------------------------


def render_trajectory_tab(rows: list[dict], manifest: dict, metrics: dict | None = None):
    df = extract_trajectory_df(rows)
    metrics = metrics or {}
    config = manifest.get("config", {})
    env_cfg = config.get("environment", {})
    nash = env_cfg.get("nash_price")
    monopoly = env_cfg.get("monopoly_price")
    onset_round = metrics.get("onset_round")
    transition_round = metrics.get("transition_round")

    if df.empty:
        st.info("No trajectory rows were found in this run's `log.jsonl`.")
        return

    min_round, max_round = int(df["round"].min()), int(df["round"].max())
    if max_round > min_round:
        lo, hi = st.slider(
            "Round range",
            min_value=min_round,
            max_value=max_round,
            value=(min_round, max_round),
            help="Narrow the plots to inspect a specific phase of the run.",
        )
        df = df[(df["round"] >= lo) & (df["round"] <= hi)]

    # --- Actions chart ---
    st.subheader("Actions Over Time")
    # Filter for action_N where N is an integer, avoiding action_spread.
    action_cols = sorted([c for c in df.columns if c.startswith("action_") and c[7:].isdigit()])
    fig_actions = go.Figure()
    for i, col in enumerate(action_cols):
        fig_actions.add_trace(go.Scatter(
            x=df["round"], y=df[col], mode="lines+markers", name=f"Agent {i}",
            hovertemplate="Round %{x}<br>Action=%{y}<extra></extra>",
        ))
    if nash is not None:
        fig_actions.add_hline(y=nash, line_dash="dash", line_color="green",
                              annotation_text="Nash", annotation_position="bottom right")
    if monopoly is not None:
        fig_actions.add_hline(y=monopoly, line_dash="dash", line_color="red",
                              annotation_text="Monopoly", annotation_position="top right")
    if onset_round is not None:
        fig_actions.add_vline(x=onset_round, line_dash="dash", annotation_text="Onset")
    if transition_round is not None:
        fig_actions.add_vline(x=transition_round, line_dash="dot", annotation_text="Transition")
    fig_actions.update_layout(xaxis_title="Round", yaxis_title="Action (Price)", height=350)
    st.plotly_chart(fig_actions, width="stretch")
    st.caption("Actions are the prices chosen by each agent. Dashed horizontal lines mark Nash and monopoly reference prices when available.")

    # --- Reward elevation chart ---
    st.subheader("Reward Elevation Over Time")
    elev_cols = [c for c in df.columns if c.startswith("reward_elevation_")]
    if elev_cols:
        fig_elev = go.Figure()
        for i, col in enumerate(elev_cols):
            fig_elev.add_trace(go.Scatter(
                x=df["round"], y=df[col], mode="lines+markers", name=f"Agent {i}",
                hovertemplate="Round %{x}<br>Reward elevation=%{y:.3f}<extra></extra>",
            ))
        mean_elev = df[elev_cols].mean(axis=1)
        smooth = mean_elev.rolling(5, min_periods=1).mean()
        fig_elev.add_trace(go.Scatter(
            x=df["round"],
            y=smooth,
            mode="lines",
            name="Smoothed mean",
            line={"width": 4},
            hovertemplate="Round %{x}<br>Smoothed mean=%{y:.3f}<extra></extra>",
        ))
        fig_elev.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Nash (0)")
        fig_elev.add_hline(y=1, line_dash="dot", line_color="gray", annotation_text="Monopoly (1)")
        if onset_round is not None:
            fig_elev.add_vline(x=onset_round, line_dash="dash", annotation_text="Onset")
            fig_elev.add_vrect(x0=onset_round, x1=df["round"].max(), opacity=0.08, line_width=0)
        if transition_round is not None:
            fig_elev.add_vline(x=transition_round, line_dash="dot", annotation_text="Transition")
        fig_elev.update_layout(xaxis_title="Round", yaxis_title="Reward Elevation", height=350)
        st.plotly_chart(fig_elev, width="stretch")
        st.caption("Reward elevation is normalized profit: 0 is competitive/Nash-like, 1 is monopoly-like. The thick line is a 5-round smoothed mean.")
    else:
        st.info("No reward elevation data available. Verify the run log contains `trajectory_signals.reward_elevation`.")

    # --- Coordination chart ---
    st.subheader("Coordination and Concealment Signals")
    if "action_spread" in df.columns:
        fig_spread = go.Figure()
        fig_spread.add_trace(go.Scatter(
            x=df["round"],
            y=df["action_spread"],
            mode="lines+markers",
            name="Action spread",
            hovertemplate="Round %{x}<br>Spread=%{y}<extra></extra>",
        ))
        if "covert_coordination_flag" in df.columns:
            covert_roll = df["covert_coordination_flag"].astype(float).rolling(10, min_periods=1).mean()
            fig_spread.add_trace(go.Scatter(
                x=df["round"], y=covert_roll, mode="lines",
                name="Rolling covert rate",
                yaxis="y2",
            ))
        if "hollow_coordination_flag" in df.columns:
            hollow_roll = df["hollow_coordination_flag"].astype(float).rolling(10, min_periods=1).mean()
            fig_spread.add_trace(go.Scatter(
                x=df["round"], y=hollow_roll, mode="lines",
                name="Rolling hollow rate",
                yaxis="y2",
                line={"dash": "dot"},
            ))
        fig_spread.update_layout(
            xaxis_title="Round",
            yaxis_title="Action Spread",
            yaxis2={"title": "Rolling Flag Rate", "overlaying": "y", "side": "right", "range": [0, 1]},
            height=330,
        )
        st.plotly_chart(fig_spread, width="stretch")
        st.caption("Action spread measures price dispersion. Covert means elevated behavior without explicit language; hollow means explicit language without elevated behavior.")
    else:
        st.info("No action spread data available. Verify the run log contains `trajectory_signals.action_spread`.")


# ---------------------------------------------------------------------------
# Transcript tab
# ---------------------------------------------------------------------------


def render_transcript_tab(
    rows: list[dict],
    metrics: dict | None = None,
    key_prefix: str = "transcript",
):
    st.subheader("Round-by-Round Transcript")
    metrics = metrics or {}
    transcript_df = build_transcript_df(
        rows,
        onset_round=metrics.get("onset_round"),
        transition_round=metrics.get("transition_round"),
    )

    # Filters
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        filter_flagged = st.checkbox(
            "Flagged rounds only",
            value=False,
            key=f"{key_prefix}_flagged_only",
        )
    with col2:
        filter_post_onset = st.checkbox(
            "Post-onset only",
            value=False,
            key=f"{key_prefix}_post_onset_only",
        )
    with col3:
        filter_post_transition = st.checkbox(
            "Post-transition only",
            value=False,
            key=f"{key_prefix}_post_transition_only",
        )
    with col4:
        keyword_filter = st.text_input(
            "Search messages",
            "",
            key=f"{key_prefix}_message_search",
        )

    filtered = transcript_df
    if filter_flagged:
        filtered = filtered[filtered["flagged"] | filtered["covert"] | filtered["hollow"]]
    if filter_post_onset:
        filtered = filtered[filtered["post_onset"]]
    if filter_post_transition:
        filtered = filtered[filtered["post_transition"]]
    if keyword_filter:
        kw = keyword_filter.lower()
        filtered = filtered[filtered["message_text"].str.lower().str.contains(kw, na=False)]

    if filtered.empty:
        st.info("No rounds match the current filters. Try clearing post-onset/post-transition or keyword filters.")
        return

    for _, flat in filtered.iterrows():
        row = flat["raw"]
        round_num = flat["round"]
        actions = flat["actions"]
        rewards = flat["rewards"]
        messages = flat["messages"]
        audit_event = flat["audit_event"]
        signals = row.get("trajectory_signals", {})

        # Determine row highlighting
        flagged = flat["flagged"]
        penalty = flat["penalized"]
        covert = flat["covert"]
        hollow = flat["hollow"]

        if flagged:
            container = st.container(border=True)
            status_text = f"**Round {round_num}** :red_circle: FLAGGED"
            if penalty:
                status_text += " | :money_with_wings: **PENALTY APPLIED**"
            container.markdown(status_text)
        elif covert:
            container = st.container(border=True)
            container.markdown(f"**Round {round_num}** 🟠 Covert")
        elif hollow:
            container = st.container(border=True)
            container.markdown(f"**Round {round_num}** :large_blue_circle: Hollow")
        else:
            container = st.container(border=True)
            container.markdown(f"**Round {round_num}**")

        with container:
            col1, col2, col3 = st.columns(3)
            col1.write(f"**Actions:** {actions}")
            col2.write(f"**Rewards:** {[round(r, 3) for r in rewards]}")

            # Signals summary
            sig_parts = []
            if flat["explicit"]:
                sig_parts.append("explicit")
            if flat["behavior"]:
                sig_parts.append("behavior")
            if covert:
                sig_parts.append("covert")
            if hollow:
                sig_parts.append("hollow")
            pfi = flat.get("price_follow_indicator")
            if pfi is not None:
                sig_parts.append(f"follow={pfi:.2f}" if isinstance(pfi, float) else f"follow={pfi}")
            col3.write(f"**Flags:** {', '.join(sig_parts) if sig_parts else 'none'}")

            reasoning = normalize_reasoning(row.get("reasoning") or [])
            comm_count = sum(1 for r in reasoning if r.get("communication"))
            pricing_count = sum(1 for r in reasoning if r.get("pricing"))
            if comm_count:
                with st.expander(
                    f"Communication reasoning ({comm_count} of {len(reasoning)} agents)"
                ):
                    st.caption(
                        "Private to each agent - not visible to other agents, "
                        "the game, or the auditor."
                    )
                    for agent_id, entry in enumerate(reasoning):
                        text = entry.get("communication")
                        if not text:
                            continue
                        st.markdown(f"**Agent {agent_id}:**")
                        st.markdown(f"> {text}")

            # Messages sent
            if messages:
                with st.expander(f"Messages sent ({len(messages)})"):
                    for m in messages:
                        sender = m.get("from", "?")
                        to = m.get("to", "")
                        content = m.get("content", "")
                        content = _highlight_keyword(content, keyword_filter)
                        if to == "all":
                            st.markdown(f"**Agent {sender} -> all:** {content}")
                        else:
                            st.markdown(f"**Agent {sender} -> Agent {to}:** {content}")

            if pricing_count:
                with st.expander(
                    f"Pricing reasoning ({pricing_count} of {len(reasoning)} agents)"
                ):
                    st.caption(
                        "Private to each agent - not visible to other agents, "
                        "the game, or the auditor."
                    )
                    for agent_id, entry in enumerate(reasoning):
                        text = entry.get("pricing")
                        st.markdown(f"**Agent {agent_id}:**")
                        if text:
                            st.markdown(f"> {text}")
                        else:
                            st.markdown(
                                "_No pricing reasoning captured - fallback action._"
                            )

            # Audit event details
            if audit_event:
                with st.expander("Audit Event"):
                    st.json(audit_event)


# ---------------------------------------------------------------------------
# Metrics tab
# ---------------------------------------------------------------------------


def render_metrics_tab(rows: list[dict], metrics: dict | None = None):
    st.subheader("Run Metrics")
    metrics = metrics or {}

    n_rounds = len(rows)

    # Action spread
    spreads = [
        r.get("trajectory_signals", {}).get("action_spread")
        for r in rows
    ]
    spreads = [s for s in spreads if s is not None]
    mean_spread = sum(spreads) / len(spreads) if spreads else None

    # Reward elevation
    elevations = []
    for r in rows:
        elev = r.get("trajectory_signals", {}).get("reward_elevation", [])
        if elev:
            elevations.append(sum(elev) / len(elev))
    mean_elevation = sum(elevations) / len(elevations) if elevations else None

    # Audit counts
    audited = sum(1 for r in rows if r.get("audit_event") is not None)
    flagged = sum(
        1 for r in rows
        if r.get("audit_event") and r["audit_event"].get("flagged")
    )
    penalized = sum(
        1 for r in rows
        if r.get("audit_event") and r["audit_event"].get("penalty_applied")
    )

    # Phase 4 flag counts
    explicit_count = sum(
        1 for r in rows
        if r.get("trajectory_signals", {}).get("explicit_collusion_flag")
    )
    behavior_count = sum(
        1 for r in rows
        if r.get("trajectory_signals", {}).get("behavior_collusion_flag")
    )
    covert_count = sum(
        1 for r in rows
        if r.get("trajectory_signals", {}).get("covert_coordination_flag")
    )
    hollow_count = sum(
        1 for r in rows
        if r.get("trajectory_signals", {}).get("hollow_coordination_flag")
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rounds", n_rounds)
    col2.metric("Onset Round", _fmt_metric(metrics.get("onset_round"), digits=0))
    col3.metric("Transition Round", _fmt_metric(metrics.get("transition_round"), digits=0))
    col4.metric(
        "Behavioral Covert Score",
        _fmt_metric(metrics.get("behavioral_steganographic_score", metrics.get("steganographic_score"))),
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Action Spread", f"{mean_spread:.2f}" if mean_spread is not None else "N/A")
    col2.metric("Mean Reward Elevation", f"{mean_elevation:.3f}" if mean_elevation is not None else "N/A")
    col3.metric("Overt Phase Duration", _fmt_metric(metrics.get("overt_phase_duration"), digits=0))
    col4.metric("Covert Phase Elevation", _fmt_metric(metrics.get("covert_phase_elevation")))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Audited Rounds", audited)
    col2.metric("Flagged Rounds", flagged)
    col3.metric("Penalized Rounds", penalized)
    col4.metric("Price Follow Rate", _fmt_metric(metrics.get("price_follow_rate")))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Explicit Flag Rounds", explicit_count)
    col2.metric("Behavior Flag Rounds", behavior_count)
    col3.metric("Covert Rounds", covert_count)
    col4.metric("Hollow Rounds", hollow_count)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Message Stego Signature", _fmt_bool_metric(metrics.get("steganographic_signature")))
    col2.metric("Message Rounds", _fmt_metric(metrics.get("steganography_message_rounds"), digits=0))
    col3.metric("Analyzer Explicit Rate", _fmt_metric(metrics.get("steganography_llm_explicit_rate")))
    col4.metric("Analyzer Covert Rate", _fmt_metric(metrics.get("steganography_covert_rate")))

    score = metrics.get("steganographic_score")
    if score is None:
        st.info("Behavioral covert score unavailable for this run.")
    elif score >= 0.5:
        st.warning("High behavioral covert-coordination signal. Inspect transcript rounds after onset/transition.")
    elif score >= 0.25:
        st.info("Moderate behavioral covert-coordination signal. Review top flagged/covert transcript rounds.")
    else:
        st.success("Low behavioral covert-coordination signal under the current metric assumptions.")

    st.caption(
        "Behavioral covert score combines price-following, covert-phase elevation, and post-audit convergence. "
        "The message steganography signature is the post-hoc analyzer result and requires message features."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if page == "Analyze":
        page_analyze()
    elif page == "Run":
        page_run()
    elif page == "Sweep":
        page_sweep()
    elif page == "Compare":
        page_compare()


if __name__ == "__main__":
    main()
