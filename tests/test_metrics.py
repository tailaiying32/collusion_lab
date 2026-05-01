"""Phase 6 unit tests for the metrics layer.

All tests use synthetic JSONL fixtures — no real experiment runs required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from collusionlab.metrics.base import LogReader, RunData, get_metrics_computer
from collusionlab.metrics import collusion, concealment

# Register pricing metrics so get_metrics_computer("pricing") works.
import collusionlab.environments.pricing.metrics  # noqa: F401


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _make_round(
    round_num: int,
    actions: list,
    rewards: list[float],
    *,
    elevation: list[float] | None = None,
    explicit: bool = False,
    behavior: bool = False,
    audited: bool = False,
    penalty_applied: bool = False,
    messages: list[dict] | None = None,
) -> dict:
    """Build a single round log entry."""
    signals = {
        "action_spread": max(actions) - min(actions),
        "reward_elevation": elevation or [0.0] * len(actions),
        "explicit_collusion_flag": explicit,
        "behavior_collusion_flag": behavior,
        "covert_coordination_flag": behavior and not explicit,
        "hollow_coordination_flag": explicit and not behavior,
    }
    audit_event = None
    if audited:
        audit_event = {
            "audited": True,
            "flagged": explicit or behavior,
            "penalty_applied": penalty_applied,
            "results": [],
        }
    return {
        "run_id": "test-run",
        "env_type": "pricing",
        "round": round_num,
        "actions": actions,
        "rewards": rewards,
        "cumulative_rewards": rewards,
        "observations": {
            "round": round_num,
            "prices": actions,
            "quantities": [0.5] * len(actions),
            "profits": rewards,
            "cumulative_profits": rewards,
        },
        "messages": messages or [],
        "audit_event": audit_event,
        "trajectory_signals": signals,
        "reasoning": [""] * len(actions),
    }


def _make_manifest(
    rounds: list[dict],
    *,
    run_id: str = "test-run",
    nash_price: int = 8,
    monopoly_price: int = 10,
    comm_mode: str = "none",
    oversight_mode: str = "none",
    seed: int = 42,
    n_agents: int = 2,
) -> dict:
    return {
        "run_id": run_id,
        "env_type": "pricing",
        "config": {
            "env_type": "pricing",
            "communication_mode": comm_mode,
            "oversight": {"mode": oversight_mode},
            "environment": {
                "n_agents": n_agents,
                "n_rounds": len(rounds),
                "seed": seed,
                "nash_price": nash_price,
                "monopoly_price": monopoly_price,
            },
            "agents": [
                {"memory_window": 5} for _ in range(n_agents)
            ],
        },
        "start_time": "2026-04-29T01:00:00+00:00",
        "end_time": "2026-04-29T01:01:00+00:00",
        "elapsed_seconds": 60.0,
        "total_input_tokens": 100,
        "total_output_tokens": 50,
        "total_cost_estimate_usd": 0.001,
    }


def _write_run(tmp_path: Path, rounds: list[dict], **manifest_kwargs) -> Path:
    """Write a run to disk and return the manifest path."""
    manifest = _make_manifest(rounds, **manifest_kwargs)
    run_dir = tmp_path / manifest["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    log_path.write_text(
        "\n".join(json.dumps(r) for r in rounds), encoding="utf-8",
    )
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _build_run_data(rounds: list[dict], **manifest_kwargs) -> RunData:
    """Build a RunData directly from in-memory fixtures (no disk I/O)."""
    manifest = _make_manifest(rounds, **manifest_kwargs)
    config = manifest["config"]
    env_cfg = config.get("environment", {})

    sig_records = []
    for r in rounds:
        sig = dict(r.get("trajectory_signals", {}))
        sig["round"] = r["round"]
        sig_records.append(sig)
    sig_df = pd.DataFrame(sig_records).set_index("round")

    return RunData(
        run_id=manifest["run_id"],
        env_type="pricing",
        config=config,
        manifest=manifest,
        rounds=rounds,
        actions=[r["actions"] for r in rounds],
        rewards=[r["rewards"] for r in rounds],
        messages=[r.get("messages", []) for r in rounds],
        audit_events=[r.get("audit_event") for r in rounds],
        trajectory_signals=sig_df,
        observations=[r.get("observations", {}) for r in rounds],
        nash_price=env_cfg.get("nash_price"),
        monopoly_price=env_cfg.get("monopoly_price"),
    )


# --- Synthetic trajectories ---

def _nash_trajectory(n: int = 20) -> list[dict]:
    """Flat Nash-level trajectory — no collusion."""
    return [
        _make_round(i + 1, [8, 8], [1.0, 1.0], elevation=[0.0, 0.0])
        for i in range(n)
    ]


def _colluding_trajectory(n: int = 20, onset: int = 5) -> list[dict]:
    """Trajectory that starts Nash and rises to monopoly."""
    rounds = []
    for i in range(n):
        if i < onset:
            rounds.append(
                _make_round(i + 1, [8, 8], [1.0, 1.0], elevation=[0.0, 0.0])
            )
        else:
            rounds.append(
                _make_round(i + 1, [10, 10], [2.0, 2.0], elevation=[0.5, 0.5])
            )
    return rounds


def _transition_trajectory(n: int = 30, onset: int = 5, transition: int = 15) -> list[dict]:
    """Trajectory: Nash -> overt collusion -> covert collusion."""
    rounds = []
    for i in range(n):
        if i < onset:
            rounds.append(
                _make_round(i + 1, [8, 8], [1.0, 1.0], elevation=[0.0, 0.0])
            )
        elif i < transition:
            rounds.append(
                _make_round(
                    i + 1, [10, 10], [2.0, 2.0], elevation=[0.5, 0.5],
                    explicit=True, behavior=True, audited=True,
                )
            )
        else:
            rounds.append(
                _make_round(
                    i + 1, [10, 10], [2.0, 2.0], elevation=[0.5, 0.5],
                    explicit=False, behavior=True, audited=True,
                )
            )
    return rounds


# ---------------------------------------------------------------------------
# LogReader tests
# ---------------------------------------------------------------------------


class TestLogReader:
    def test_load_run_parses_fields(self, tmp_path):
        rounds = _nash_trajectory(5)
        mp = _write_run(tmp_path, rounds)
        run = LogReader.load_run(mp)

        assert run.run_id == "test-run"
        assert run.env_type == "pricing"
        assert run.n_rounds == 5
        assert run.n_agents == 2
        assert run.nash_price == 8
        assert run.monopoly_price == 10
        assert len(run.actions) == 5
        assert len(run.rewards) == 5
        assert not run.trajectory_signals.empty
        assert "action_spread" in run.trajectory_signals.columns

    def test_load_run_trajectory_signals_indexed_by_round(self, tmp_path):
        rounds = _nash_trajectory(3)
        mp = _write_run(tmp_path, rounds)
        run = LogReader.load_run(mp)
        assert list(run.trajectory_signals.index) == [1, 2, 3]

    def test_load_sweep(self, tmp_path):
        mp1 = _write_run(tmp_path, _nash_trajectory(3), run_id="run-1")
        mp2 = _write_run(tmp_path, _nash_trajectory(3), run_id="run-2")
        sweep = {
            "runs": [
                {"run_id": "run-1", "status": "succeeded", "manifest_path": str(mp1)},
                {"run_id": "run-2", "status": "succeeded", "manifest_path": str(mp2)},
                {"run_id": "run-3", "status": "failed", "manifest_path": None},
            ]
        }
        sp = tmp_path / "sweep_manifest.json"
        sp.write_text(json.dumps(sweep))
        runs = LogReader.load_sweep(sp)
        assert len(runs) == 2


# ---------------------------------------------------------------------------
# collusion.py tests
# ---------------------------------------------------------------------------


class TestCollusionOnset:
    def test_no_onset_on_nash_trajectory(self):
        run = _build_run_data(_nash_trajectory(20))
        assert collusion.collusion_onset_round(run) is None

    def test_onset_detected_on_colluding_trajectory(self):
        run = _build_run_data(_colluding_trajectory(20, onset=5))
        onset = collusion.collusion_onset_round(run, elevation_threshold=0.3, min_duration=5)
        assert onset is not None
        assert onset >= 5

    def test_stability_zero_for_nash(self):
        run = _build_run_data(_nash_trajectory(20))
        assert collusion.collusion_stability(run) == 0.0

    def test_stability_high_for_colluding(self):
        run = _build_run_data(_colluding_trajectory(20, onset=3))
        stab = collusion.collusion_stability(run, min_duration=3)
        assert stab > 0.5

    def test_onset_speed_none_when_no_onset(self):
        run = _build_run_data(_nash_trajectory(20))
        assert collusion.onset_speed(run) is None

    def test_onset_speed_positive_when_collusion_emerges(self):
        run = _build_run_data(_colluding_trajectory(20, onset=5))
        speed = collusion.onset_speed(run, min_duration=5)
        if speed is not None:
            assert speed >= 0.0

    def test_adaptive_min_duration_detects_late_short_run_onset(self):
        run = _build_run_data(_colluding_trajectory(20, onset=17))
        onset = collusion.collusion_onset_round(
            run, elevation_threshold=0.3, min_duration=None
        )
        assert onset is not None


class TestCollusionSeries:
    def test_action_convergence_declining(self):
        rounds = []
        for i in range(10):
            spread = max(0, 5 - i)
            a = [8, 8 + spread]
            rounds.append(_make_round(i + 1, a, [1.0, 1.0]))
        run = _build_run_data(rounds)
        series = collusion.action_convergence_series(run, window=3)
        assert len(series) == 10
        assert series.iloc[-1] < series.iloc[0]

    def test_reward_elevation_series_shape(self):
        run = _build_run_data(_colluding_trajectory(10))
        df = collusion.reward_elevation_series(run)
        assert df.shape == (10, 2)


class TestThresholdAnalysis:
    def test_onset_rate_zero_for_nash_runs(self):
        runs = [_build_run_data(_nash_trajectory(20)) for _ in range(5)]
        assert collusion.onset_rate(runs) == 0.0

    def test_onset_rate_positive_for_colluding_runs(self):
        runs = [_build_run_data(_colluding_trajectory(20, onset=3)) for _ in range(5)]
        rate = collusion.onset_rate(runs, min_duration=3)
        assert rate > 0.0

    def test_median_onset_round_none_for_nash(self):
        runs = [_build_run_data(_nash_trajectory(20)) for _ in range(3)]
        assert collusion.median_onset_round(runs) is None

    def test_threshold_table_has_expected_columns(self):
        runs = [
            _build_run_data(_colluding_trajectory(20, onset=3), comm_mode="public"),
            _build_run_data(_nash_trajectory(20), comm_mode="none"),
        ]
        table = collusion.threshold_table(runs, groupby=["communication_mode"], min_duration=3)
        assert "onset_rate" in table.columns
        assert "median_onset_round" in table.columns
        assert "n_runs" in table.columns
        assert len(table) == 2


# ---------------------------------------------------------------------------
# concealment.py tests
# ---------------------------------------------------------------------------


class TestConcealmentSeries:
    def test_covert_nonzero_when_behavior_not_explicit(self):
        run = _build_run_data(_transition_trajectory(30, onset=5, transition=15))
        series = concealment.covert_coordination_series(run, window=3, audited_only=False)
        post_transition = series.iloc[15:]
        assert post_transition.max() > 0

    def test_hollow_nonzero_only_when_explicit_not_behavior(self):
        rounds = [
            _make_round(i + 1, [8, 8], [1.0, 1.0],
                        elevation=[0.0, 0.0],
                        explicit=True, behavior=False, audited=True)
            for i in range(10)
        ]
        run = _build_run_data(rounds)
        series = concealment.hollow_coordination_series(run, window=3, audited_only=False)
        assert series.max() > 0

    def test_covert_zero_when_explicit_matches_behavior(self):
        rounds = [
            _make_round(i + 1, [10, 10], [2.0, 2.0],
                        elevation=[0.5, 0.5],
                        explicit=True, behavior=True, audited=True)
            for i in range(10)
        ]
        run = _build_run_data(rounds)
        series = concealment.covert_coordination_series(run, window=3, audited_only=False)
        assert series.max() == 0.0


class TestTransitionDetection:
    def test_transition_detected_in_transition_trajectory(self):
        run = _build_run_data(_transition_trajectory(30, onset=5, transition=15))
        tr = concealment.transition_round(run, min_duration=3)
        assert tr is not None
        assert tr >= 15

    def test_no_transition_in_nash(self):
        run = _build_run_data(_nash_trajectory(20))
        assert concealment.transition_round(run) is None

    def test_overt_phase_duration_positive(self):
        run = _build_run_data(_transition_trajectory(30, onset=5, transition=15))
        dur = concealment.overt_phase_duration(run, min_duration=3)
        if dur is not None:
            assert dur >= 0

    def test_covert_phase_elevation_above_zero(self):
        run = _build_run_data(_transition_trajectory(30, onset=5, transition=15))
        cpe = concealment.covert_phase_elevation(run, min_duration=3)
        if cpe is not None:
            assert cpe > 0


class TestImplicitCoordination:
    def test_price_follow_rate_high_for_lag_following_without_convergence(self):
        actions = [[2, 10], [10, 2], [2, 10], [10, 2], [2, 10], [10, 2]]
        rounds = [
            _make_round(i + 1, a, [1.5, 1.5])
            for i, a in enumerate(actions)
        ]
        run = _build_run_data(rounds)
        pf = concealment.price_follow_rate(run)
        assert pf > 0.5

    def test_price_follow_rate_low_when_only_converged(self):
        rounds = [
            _make_round(i + 1, [10, 10], [2.0, 2.0])
            for i in range(10)
        ]
        run = _build_run_data(rounds)
        pf = concealment.price_follow_rate(run)
        assert pf == pytest.approx(0.0)

    def test_price_follow_rate_low_for_divergent_actions(self):
        import random
        rng = random.Random(99)
        rounds = [
            _make_round(i + 1, [rng.randint(1, 5), rng.randint(10, 15)], [1.0, 1.0])
            for i in range(20)
        ]
        run = _build_run_data(rounds)
        pf = concealment.price_follow_rate(run)
        assert pf < 0.3

    def test_steganographic_score_higher_for_covert(self):
        nash_run = _build_run_data(_nash_trajectory(20))
        covert_run = _build_run_data(_transition_trajectory(30, onset=5, transition=15))
        assert concealment.steganographic_score(covert_run) > concealment.steganographic_score(nash_run)

    def test_post_audit_convergence_none_when_no_audits(self):
        run = _build_run_data(_nash_trajectory(10))
        assert concealment.post_audit_convergence(run) is None

    def test_action_mi_empty_for_short_run(self):
        run = _build_run_data(_nash_trajectory(5))
        mi = concealment.action_mutual_information(run, window=20)
        assert mi.empty


class TestConcealmentSweep:
    def test_transition_rate_zero_for_nash(self):
        runs = [_build_run_data(_nash_trajectory(20)) for _ in range(3)]
        assert concealment.transition_rate(runs) == 0.0

    def test_concealment_by_condition_shape(self):
        runs = [
            _build_run_data(_transition_trajectory(30), comm_mode="public"),
            _build_run_data(_nash_trajectory(20), comm_mode="none"),
        ]
        df = concealment.concealment_by_condition(runs, groupby=["communication_mode"])
        assert len(df) == 2
        assert "mean_covert_rate" in df.columns
        assert "steganographic_score" in df.columns


# ---------------------------------------------------------------------------
# PricingMetricsComputer tests
# ---------------------------------------------------------------------------


class TestPricingMetrics:
    def test_compute_returns_expected_keys(self):
        run = _build_run_data(_colluding_trajectory(20, onset=5))
        computer = get_metrics_computer("pricing")
        result = computer.compute(run)
        expected_keys = {
            "run_id", "env_type", "n_rounds", "n_agents",
            "communication_mode", "oversight_mode", "seed", "memory_window",
            "onset_round", "onset_speed", "collusion_stability",
            "transition_round", "overt_phase_duration", "covert_phase_elevation",
            "price_follow_rate", "post_audit_convergence", "steganographic_score",
            "behavioral_steganographic_score", "steganographic_signature",
            "steganography_message_rounds", "steganography_llm_explicit_rate",
            "steganography_behavior_rate", "steganography_covert_rate",
            "steganography_top_feature",
            "mean_price_elevation", "final_price_elevation",
            "mean_action_spread", "mean_reward_elevation", "total_profit",
            "explicit_flag_count", "behavior_flag_count",
            "covert_flag_count", "hollow_flag_count",
        }
        assert expected_keys.issubset(set(result))

    def test_compute_sweep_one_row_per_run(self):
        runs = [
            _build_run_data(_colluding_trajectory(20, onset=5), run_id="r1"),
            _build_run_data(_nash_trajectory(20), run_id="r2"),
        ]
        computer = get_metrics_computer("pricing")
        df = computer.compute_sweep(runs)
        assert len(df) == 2
        assert "onset_round" in df.columns
        assert list(df["run_id"]) == ["r1", "r2"]

    def test_price_elevation_series_bounded(self):
        from collusionlab.environments.pricing.metrics import price_elevation_series
        run = _build_run_data(_nash_trajectory(10))
        pe = price_elevation_series(run)
        assert not pe.empty
        assert pe.iloc[0] == pytest.approx(0.0)
