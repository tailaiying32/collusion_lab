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


def _safe_run_metrics(run_dir: Path) -> tuple[dict, object | None]:
    """Load RunData + computed metrics for a run directory."""
    try:
        run_data = LogReader.load_run(run_dir / "manifest.json")
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

    options = {
        f"{s['sweep_id'][:8]}... | {s['started_at'][:19]} | {s['mode']} | runs={s['n_runs']}": s["path"]
        for s in sweeps
    }
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

    tab_threshold, tab_transition, tab_browser = st.tabs(
        ["Threshold View", "Transition View", "Run Browser"]
    )

    with tab_threshold:
        render_threshold_view(runs, sweep_df)
    with tab_transition:
        render_transition_view(runs, sweep_df)
    with tab_browser:
        render_run_browser(sweep_df, run_paths)


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
            title="Steganographic Score by Condition",
        )
        st.plotly_chart(fig, width="stretch")

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
        min_score = st.number_input("Min steganographic score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
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
        selected_run = st.selectbox("Open run in Analyze", selectable)
        if st.button("Open selected run"):
            st.session_state["_selected_run_id"] = selected_run
            st.session_state["_nav_target"] = "Analyze"
            st.rerun()
    else:
        st.caption("Run jump is unavailable because sweep manifest paths were not found.")


# ---------------------------------------------------------------------------
# Analyze page
# ---------------------------------------------------------------------------


def page_analyze():
    st.header("Analyze Run")

    raw_dir = Path("data/raw")
    if not raw_dir.exists():
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

    preselected = st.session_state.pop("_selected_run_id", None)
    labels = list(filtered["label"])
    default_index = 0
    if preselected:
        matches = filtered.index[filtered["run_id"] == preselected].tolist()
        if matches:
            default_index = list(filtered.index).index(matches[0])

    selected_label = st.selectbox(
        "Select run",
        options=labels,
        index=default_index,
    )
    selected_row = filtered[filtered["label"] == selected_label].iloc[0]
    run_dir = Path(selected_row["run_dir"])

    manifest = load_manifest(run_dir)
    rows = load_log_rows(run_dir)

    if manifest is None or not rows:
        st.error("Failed to load run data.")
        return
    metrics, run_data = _safe_run_metrics(run_dir)

    # --- Tabs ---
    tab_config, tab_trajectory, tab_transcript, tab_metrics = st.tabs(
        ["Config", "Trajectory", "Transcript", "Metrics"]
    )

    with tab_config:
        render_config_tab(manifest, run_index, run_dir)

    with tab_trajectory:
        render_trajectory_tab(rows, manifest, metrics)

    with tab_transcript:
        render_transcript_tab(rows, metrics)

    with tab_metrics:
        render_metrics_tab(rows, metrics)


# ---------------------------------------------------------------------------
# Config tab
# ---------------------------------------------------------------------------


def render_config_tab(manifest: dict, run_index: pd.DataFrame | None = None, current_run_dir: Path | None = None):
    st.subheader("Run Metadata")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Elapsed (s)", f"{manifest.get('elapsed_seconds', 0):.1f}")
    col2.metric("Input Tokens", f"{manifest.get('total_input_tokens', 0):,}")
    col3.metric("Output Tokens", f"{manifest.get('total_output_tokens', 0):,}")
    col4.metric("Est. Cost ($)", f"{manifest.get('total_cost_estimate_usd', 0):.4f}")

    st.subheader("Configuration")
    config = manifest.get("config", {})
    st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language="yaml")

    if run_index is not None and len(run_index) > 1:
        with st.expander("Compare config against another run"):
            options = {
                row["label"]: Path(row["run_dir"])
                for _, row in run_index.iterrows()
                if current_run_dir is None or Path(row["run_dir"]) != current_run_dir
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


def render_transcript_tab(rows: list[dict], metrics: dict | None = None):
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
        filter_flagged = st.checkbox("Flagged rounds only", value=False)
    with col2:
        filter_post_onset = st.checkbox("Post-onset only", value=False)
    with col3:
        filter_post_transition = st.checkbox("Post-transition only", value=False)
    with col4:
        keyword_filter = st.text_input("Search messages", "")

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

            # Messages
            if messages:
                with st.expander(f"Messages ({len(messages)})"):
                    for m in messages:
                        sender = m.get("from", "?")
                        content = m.get("content", "")
                        content = _highlight_keyword(content, keyword_filter)
                        st.markdown(f"**Agent {sender}:** {content}")

            reasoning = row.get("reasoning") or []
            if any(r for r in reasoning):
                n_non_empty = sum(1 for r in reasoning if r)
                with st.expander(
                    f"Internal reasoning ({n_non_empty} of {len(reasoning)} agents)"
                ):
                    st.caption(
                        "Private to each agent — not visible to other agents, "
                        "the game, or the auditor."
                    )
                    for agent_id, text in enumerate(reasoning):
                        st.markdown(f"**Agent {agent_id}:**")
                        if text:
                            st.markdown(f"> {text}")
                        else:
                            st.markdown(
                                "_No reasoning captured — fallback action._"
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
    col4.metric("Steganographic Score", _fmt_metric(metrics.get("steganographic_score")))

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

    score = metrics.get("steganographic_score")
    if score is None:
        st.info("Steganographic score unavailable for this run.")
    elif score >= 0.5:
        st.warning("High covert-coordination signal. Inspect transcript rounds after onset/transition.")
    elif score >= 0.25:
        st.info("Moderate covert-coordination signal. Review top flagged/covert transcript rounds.")
    else:
        st.success("Low covert-coordination signal under the current metric assumptions.")

    st.caption(
        "Steganographic score combines price-following, covert-phase elevation, and post-audit convergence. "
        "It is a triage signal, not a ground-truth label."
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
