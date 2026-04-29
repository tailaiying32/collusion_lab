"""CollusionLab Analysis UI.

Streamlit app for inspecting experiment runs, launching experiments and
parameter sweeps, and comparing results across conditions.

Launch:
    streamlit run src/collusionlab/ui/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from collusionlab.ui.data_loading import (
    list_runs,
    load_log_rows,
    load_manifest,
    extract_trajectory_df,
)
from collusionlab.ui.run_page import render_run_page
from collusionlab.ui.sweep_page import render_sweep_page

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

_PAGES = ["Analyze", "Run", "Sweep", "Compare"]
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


def page_compare():
    st.header("Compare Runs")
    st.info("Coming soon: cross-run comparison, threshold tables, and sweep analysis.")


# ---------------------------------------------------------------------------
# Analyze page
# ---------------------------------------------------------------------------


def page_analyze():
    st.header("Analyze Run")

    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        st.warning("No runs found. `data/raw/` does not exist.")
        return

    runs = list_runs(raw_dir)
    if not runs:
        st.warning("No runs found in `data/raw/`.")
        return

    # --- Run selector ---
    run_options = {
        f"{r['run_id'][:8]}... | {r['start_time'][:19]} | {r['comm_mode']} | {r['oversight_mode']}": r["run_dir"]
        for r in runs
    }
    selected_label = st.selectbox(
        "Select run",
        options=list(run_options.keys()),
        index=0,
    )
    run_dir = Path(run_options[selected_label])

    manifest = load_manifest(run_dir)
    rows = load_log_rows(run_dir)

    if manifest is None or not rows:
        st.error("Failed to load run data.")
        return

    # --- Tabs ---
    tab_config, tab_trajectory, tab_transcript, tab_metrics = st.tabs(
        ["Config", "Trajectory", "Transcript", "Metrics"]
    )

    with tab_config:
        render_config_tab(manifest)

    with tab_trajectory:
        render_trajectory_tab(rows, manifest)

    with tab_transcript:
        render_transcript_tab(rows)

    with tab_metrics:
        render_metrics_tab(rows)


# ---------------------------------------------------------------------------
# Config tab
# ---------------------------------------------------------------------------


def render_config_tab(manifest: dict):
    st.subheader("Run Metadata")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Elapsed (s)", f"{manifest.get('elapsed_seconds', 0):.1f}")
    col2.metric("Input Tokens", f"{manifest.get('total_input_tokens', 0):,}")
    col3.metric("Output Tokens", f"{manifest.get('total_output_tokens', 0):,}")
    col4.metric("Est. Cost ($)", f"{manifest.get('total_cost_estimate_usd', 0):.4f}")

    st.subheader("Configuration")
    config = manifest.get("config", {})
    st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language="yaml")


# ---------------------------------------------------------------------------
# Trajectory tab
# ---------------------------------------------------------------------------


def render_trajectory_tab(rows: list[dict], manifest: dict):
    df = extract_trajectory_df(rows)
    config = manifest.get("config", {})
    env_cfg = config.get("environment", {})
    nash = env_cfg.get("nash_price")
    monopoly = env_cfg.get("monopoly_price")
    n_agents = env_cfg.get("n_agents", 2)

    # --- Actions chart ---
    st.subheader("Actions Over Time")
    # Filter for action_N where N is an integer, avoiding action_spread.
    action_cols = sorted([c for c in df.columns if c.startswith("action_") and c[7:].isdigit()])
    fig_actions = go.Figure()
    for i, col in enumerate(action_cols):
        fig_actions.add_trace(go.Scatter(
            x=df["round"], y=df[col], mode="lines+markers", name=f"Agent {i}"
        ))
    if nash is not None:
        fig_actions.add_hline(y=nash, line_dash="dash", line_color="green",
                              annotation_text="Nash", annotation_position="bottom right")
    if monopoly is not None:
        fig_actions.add_hline(y=monopoly, line_dash="dash", line_color="red",
                              annotation_text="Monopoly", annotation_position="top right")
    fig_actions.update_layout(xaxis_title="Round", yaxis_title="Action (Price)", height=350)
    st.plotly_chart(fig_actions, width="stretch")

    # --- Reward elevation chart ---
    st.subheader("Reward Elevation Over Time")
    elev_cols = [c for c in df.columns if c.startswith("reward_elevation_")]
    if elev_cols:
        fig_elev = go.Figure()
        for i, col in enumerate(elev_cols):
            fig_elev.add_trace(go.Scatter(
                x=df["round"], y=df[col], mode="lines+markers", name=f"Agent {i}"
            ))
        fig_elev.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Nash (0)")
        fig_elev.add_hline(y=1, line_dash="dot", line_color="gray", annotation_text="Monopoly (1)")
        fig_elev.update_layout(xaxis_title="Round", yaxis_title="Reward Elevation", height=350)
        st.plotly_chart(fig_elev, width="stretch")
    else:
        st.info("No reward elevation data available.")

    # --- Action spread chart ---
    st.subheader("Action Spread Over Time")
    if "action_spread" in df.columns:
        fig_spread = px.line(df, x="round", y="action_spread", markers=True)
        fig_spread.update_layout(xaxis_title="Round", yaxis_title="Action Spread", height=300)
        st.plotly_chart(fig_spread, width="stretch")
    else:
        st.info("No action spread data available.")


# ---------------------------------------------------------------------------
# Transcript tab
# ---------------------------------------------------------------------------


def render_transcript_tab(rows: list[dict]):
    st.subheader("Round-by-Round Transcript")

    # Filters
    col1, col2 = st.columns([1, 2])
    with col1:
        filter_flagged = st.checkbox("Flagged rounds only", value=False)
    with col2:
        keyword_filter = st.text_input("Search messages", "")

    filtered_rows = rows
    if filter_flagged:
        filtered_rows = [
            r for r in filtered_rows
            if r.get("audit_event") and r["audit_event"].get("flagged")
        ]
    if keyword_filter:
        kw = keyword_filter.lower()
        filtered_rows = [
            r for r in filtered_rows
            if any(kw in m.get("content", "").lower() for m in r.get("messages", []))
        ]

    if not filtered_rows:
        st.info("No rounds match the current filters.")
        return

    for row in filtered_rows:
        round_num = row.get("round", "?")
        actions = row.get("actions", [])
        rewards = row.get("rewards", [])
        messages = row.get("messages", [])
        audit_event = row.get("audit_event")
        signals = row.get("trajectory_signals", {})

        # Determine row highlighting
        flagged = audit_event and audit_event.get("flagged")
        penalty = audit_event and audit_event.get("penalty_applied")
        covert = signals.get("covert_coordination_flag", False)
        hollow = signals.get("hollow_coordination_flag", False)

        if flagged:
            container = st.container(border=True)
            status_text = f"**Round {round_num}** :red_circle: FLAGGED"
            if penalty:
                status_text += " | :money_with_wings: **PENALTY APPLIED**"
            container.markdown(status_text)
        elif covert:
            container = st.container(border=True)
            container.markdown(f"**Round {round_num}** :large_orange_circle: Covert")
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
            if signals.get("explicit_collusion_flag"):
                sig_parts.append("explicit")
            if signals.get("behavior_collusion_flag"):
                sig_parts.append("behavior")
            if covert:
                sig_parts.append("covert")
            if hollow:
                sig_parts.append("hollow")
            col3.write(f"**Flags:** {', '.join(sig_parts) if sig_parts else 'none'}")

            # Messages
            if messages:
                with st.expander(f"Messages ({len(messages)})"):
                    for m in messages:
                        sender = m.get("from", "?")
                        content = m.get("content", "")
                        if keyword_filter and keyword_filter.lower() in content.lower():
                            content = content.replace(
                                keyword_filter,
                                f"**{keyword_filter}**"
                            )
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


def render_metrics_tab(rows: list[dict]):
    st.subheader("Quick Metrics")

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
    col2.metric("Mean Action Spread", f"{mean_spread:.2f}" if mean_spread is not None else "N/A")
    col3.metric("Mean Reward Elevation", f"{mean_elevation:.3f}" if mean_elevation is not None else "N/A")
    col4.metric("Audited Rounds", audited)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Flagged Rounds", flagged)
    col2.metric("Penalized Rounds", penalized)
    col3.metric("Explicit Flag Rounds", explicit_count)
    col4.metric("Behavior Flag Rounds", behavior_count)

    col1, col2 = st.columns(2)
    col1.metric("Covert Rounds", covert_count)
    col2.metric("Hollow Rounds", hollow_count)


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
