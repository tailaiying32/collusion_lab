"""Tests for UI data loading helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from collusionlab.ui.data_loading import (
    build_compare_df,
    build_run_index,
    build_transcript_df,
    extract_trajectory_df,
    get_recent_config,
    get_signal,
    list_sweeps,
    list_runs,
    load_log_rows,
    load_manifest,
    load_recent_configs,
    load_sweep_manifest,
    normalize_reasoning,
    set_recent_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_manifest() -> dict:
    return {
        "run_id": "test-run-123",
        "env_type": "pricing",
        "start_time": "2026-04-29T01:00:00+00:00",
        "end_time": "2026-04-29T01:01:00+00:00",
        "elapsed_seconds": 60.0,
        "total_input_tokens": 1000,
        "total_output_tokens": 500,
        "total_cost_estimate_usd": 0.01,
        "config": {
            "env_type": "pricing",
            "communication_mode": "public",
            "oversight": {"mode": "audit-penalty"},
            "environment": {
                "n_agents": 2,
                "n_rounds": 5,
                "nash_price": 8,
                "monopoly_price": 10,
            },
        },
    }


@pytest.fixture
def sample_log_rows() -> list[dict]:
    return [
        {
            "round": 1,
            "actions": [8, 9],
            "rewards": [1.5, 1.2],
            "messages": [{"from": 0, "to": "all", "content": "Hello"}],
            "audit_event": None,
            "trajectory_signals": {
                "action_spread": 1,
                "reward_elevation": [0.2, 0.1],
                "explicit_collusion_flag": False,
                "behavior_collusion_flag": False,
                "covert_coordination_flag": False,
                "hollow_coordination_flag": False,
            },
        },
        {
            "round": 2,
            "actions": [10, 10],
            "rewards": [2.0, 2.0],
            "messages": [],
            "audit_event": {"audited": True, "flagged": True, "penalty_applied": True, "results": []},
            "trajectory_signals": {
                "action_spread": 0,
                "reward_elevation": [1.0, 1.0],
                "explicit_collusion_flag": True,
                "behavior_collusion_flag": True,
                "covert_coordination_flag": False,
                "hollow_coordination_flag": False,
            },
        },
    ]


# ---------------------------------------------------------------------------
# list_runs tests
# ---------------------------------------------------------------------------


def test_list_runs_returns_empty_for_nonexistent_dir(tmp_path):
    result = list_runs(tmp_path / "does_not_exist")
    assert result == []


def test_list_runs_returns_empty_for_empty_dir(tmp_path):
    result = list_runs(tmp_path)
    assert result == []


def test_list_runs_discovers_runs_sorted_by_start_time(tmp_path, sample_manifest):
    # Create two runs with different start times
    run1 = tmp_path / "run-older"
    run1.mkdir()
    manifest1 = {**sample_manifest, "run_id": "run-older", "start_time": "2026-04-28T01:00:00+00:00"}
    (run1 / "manifest.json").write_text(json.dumps(manifest1))

    run2 = tmp_path / "run-newer"
    run2.mkdir()
    manifest2 = {**sample_manifest, "run_id": "run-newer", "start_time": "2026-04-29T01:00:00+00:00"}
    (run2 / "manifest.json").write_text(json.dumps(manifest2))

    runs = list_runs(tmp_path)
    assert len(runs) == 2
    assert runs[0]["run_id"] == "run-newer"  # newest first
    assert runs[1]["run_id"] == "run-older"


def test_list_runs_extracts_metadata(tmp_path, sample_manifest):
    run_dir = tmp_path / "test-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(sample_manifest))

    runs = list_runs(tmp_path)
    assert len(runs) == 1
    r = runs[0]
    assert r["run_id"] == "test-run-123"
    assert r["env_type"] == "pricing"
    assert r["comm_mode"] == "public"
    assert r["oversight_mode"] == "audit-penalty"
    assert r["run_dir"] == run_dir


def test_list_runs_skips_corrupt_manifest(tmp_path):
    run_dir = tmp_path / "corrupt-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("not valid json {{{")

    runs = list_runs(tmp_path)
    assert runs == []


def test_list_sweeps_returns_empty_for_nonexistent_dir(tmp_path):
    result = list_sweeps(tmp_path / "does_not_exist")
    assert result == []


def test_list_sweeps_discovers_sweeps_sorted_by_started_at(tmp_path):
    s1 = tmp_path / "sweep_a"
    s1.mkdir()
    (s1 / "sweep_manifest.json").write_text(json.dumps({
        "sweep_id": "a",
        "started_at": "2026-04-28T01:00:00+00:00",
        "mode": "grid",
        "runs": [1],
    }))

    s2 = tmp_path / "sweep_b"
    s2.mkdir()
    (s2 / "sweep_manifest.json").write_text(json.dumps({
        "sweep_id": "b",
        "started_at": "2026-04-29T01:00:00+00:00",
        "mode": "list",
        "runs": [1, 2],
    }))

    sweeps = list_sweeps(tmp_path)
    assert len(sweeps) == 2
    assert sweeps[0]["sweep_id"] == "b"
    assert sweeps[1]["sweep_id"] == "a"


def test_load_sweep_manifest_success(tmp_path):
    path = tmp_path / "sweep_manifest.json"
    path.write_text(json.dumps({"sweep_id": "x"}))
    data = load_sweep_manifest(path)
    assert data is not None
    assert data["sweep_id"] == "x"


def test_load_sweep_manifest_missing_or_corrupt(tmp_path):
    assert load_sweep_manifest(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert load_sweep_manifest(bad) is None


# ---------------------------------------------------------------------------
# load_manifest tests
# ---------------------------------------------------------------------------


def test_load_manifest_success(tmp_path, sample_manifest):
    run_dir = tmp_path / "test-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(sample_manifest))

    manifest = load_manifest(run_dir)
    assert manifest is not None
    assert manifest["run_id"] == "test-run-123"


def test_load_manifest_returns_none_for_missing(tmp_path):
    assert load_manifest(tmp_path / "nonexistent") is None


def test_load_manifest_returns_none_for_corrupt(tmp_path):
    run_dir = tmp_path / "corrupt"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("invalid")
    assert load_manifest(run_dir) is None


# ---------------------------------------------------------------------------
# load_log_rows tests
# ---------------------------------------------------------------------------


def test_load_log_rows_success(tmp_path, sample_log_rows):
    run_dir = tmp_path / "test-run"
    run_dir.mkdir()
    log_content = "\n".join(json.dumps(r) for r in sample_log_rows)
    (run_dir / "log.jsonl").write_text(log_content)

    rows = load_log_rows(run_dir)
    assert len(rows) == 2
    assert rows[0]["round"] == 1
    assert rows[1]["round"] == 2


def test_load_log_rows_returns_empty_for_missing(tmp_path):
    assert load_log_rows(tmp_path / "nonexistent") == []


def test_load_log_rows_handles_empty_lines(tmp_path, sample_log_rows):
    run_dir = tmp_path / "test-run"
    run_dir.mkdir()
    log_content = json.dumps(sample_log_rows[0]) + "\n\n" + json.dumps(sample_log_rows[1]) + "\n"
    (run_dir / "log.jsonl").write_text(log_content)

    rows = load_log_rows(run_dir)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# extract_trajectory_df tests
# ---------------------------------------------------------------------------


def test_extract_trajectory_df_basic(sample_log_rows):
    df = extract_trajectory_df(sample_log_rows)
    assert len(df) == 2
    assert list(df["round"]) == [1, 2]
    assert "action_0" in df.columns
    assert "action_1" in df.columns
    assert "reward_elevation_0" in df.columns
    assert "action_spread" in df.columns


def test_extract_trajectory_df_extracts_flags(sample_log_rows):
    df = extract_trajectory_df(sample_log_rows)
    assert df.loc[0, "explicit_collusion_flag"] == False
    assert df.loc[1, "explicit_collusion_flag"] == True
    assert df.loc[1, "behavior_collusion_flag"] == True


def test_extract_trajectory_df_handles_empty_list():
    df = extract_trajectory_df([])
    assert len(df) == 0


def test_extract_trajectory_df_handles_missing_signals():
    rows = [{"round": 1, "actions": [5], "rewards": [1.0], "trajectory_signals": {}}]
    df = extract_trajectory_df(rows)
    assert len(df) == 1
    assert df.loc[0, "action_spread"] is None


def test_get_signal_price_follow_fallback():
    row = {"trajectory_signals": {"price_follow_lag1": 0.5}}
    assert get_signal(row, "price_follow_indicator") == 0.5
    assert get_signal(row, "missing", "x") == "x"


def test_normalize_reasoning_handles_new_and_legacy_shapes():
    assert normalize_reasoning([
        {"communication": "comm", "pricing": "price"},
        "legacy price",
    ]) == [
        {"communication": "comm", "pricing": "price"},
        {"communication": None, "pricing": "legacy price"},
    ]


def test_build_run_index_adds_label_and_date(tmp_path, sample_manifest):
    run_dir = tmp_path / "test-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(sample_manifest))
    df = build_run_index(tmp_path)
    assert len(df) == 1
    assert "label" in df.columns
    assert "rounds=5" in df.loc[0, "label"]
    assert "oversight=audit-penalty" in df.loc[0, "label"]
    assert df.loc[0, "date"] is not None


def test_build_transcript_df_derives_filters(sample_log_rows):
    df = build_transcript_df(sample_log_rows, onset_round=2, transition_round=2)
    assert list(df["round"]) == [1, 2]
    assert df.loc[0, "post_onset"] == False
    assert df.loc[1, "post_onset"] == True
    assert df.loc[1, "flagged"] == True


def test_build_compare_df_adds_concealment_gap():
    import pandas as pd
    df = build_compare_df(pd.DataFrame([{
        "run_id": "r",
        "n_rounds": 10,
        "covert_flag_count": 3,
        "hollow_flag_count": 1,
        "onset_round": 2,
        "transition_round": None,
    }]))
    assert df.loc[0, "mean_covert_rate"] == 0.3
    assert df.loc[0, "mean_hollow_rate"] == 0.1
    assert df.loc[0, "concealment_gap"] == pytest.approx(0.2)
    assert df.loc[0, "has_onset"] == True
    assert df.loc[0, "has_transition"] == False


def test_recent_config_persistence_round_trip(tmp_path, monkeypatch):
    from collusionlab.ui import data_loading

    monkeypatch.setattr(data_loading, "UI_PREFS_PATH", tmp_path / "recent.json")
    assert load_recent_configs() == {}
    set_recent_config("run_page_last_config", "base.yaml")
    set_recent_config("sweep_page_last_base_config", "pricing_audit.yaml")
    assert get_recent_config("run_page_last_config") == "base.yaml"
    prefs = load_recent_configs()
    assert prefs["run_page_last_config"] == "base.yaml"
    assert prefs["sweep_page_last_base_config"] == "pricing_audit.yaml"
