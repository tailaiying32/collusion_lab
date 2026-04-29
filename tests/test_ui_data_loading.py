"""Tests for UI data loading helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from collusionlab.ui.data_loading import (
    extract_trajectory_df,
    list_runs,
    load_log_rows,
    load_manifest,
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
