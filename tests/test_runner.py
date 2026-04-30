"""Phase 3 unit tests for the experiment runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Importing pricing registers the env; importing scripted_client registers the backend.
from collusionlab.environments.pricing import PricingConfig, PricingGame  # noqa: F401
import collusionlab.agents.backends.scripted_client  # noqa: F401  (registers "scripted")
from collusionlab.agents.model_client import ModelClient, register_backend
from collusionlab.environments.base import get_environment
from collusionlab.runner.config import ExperimentConfig
from collusionlab.runner.experiment import Experiment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_env_block() -> dict:
    import yaml
    with (ROOT / "configs" / "base.yaml").open() as f:
        cfg = yaml.safe_load(f)
    env = cfg["environment"]
    env.pop("_calibration_note", None)
    return env


def _make_config(
    *,
    n_rounds: int = 5,
    seed: int = 0,
    run_id: str | None = "test-run",
    output_dir: str,
    replies_per_agent: list[list[str]],
    comm_mode: str = "none",
) -> ExperimentConfig:
    env = _base_env_block()
    env["n_rounds"] = n_rounds
    env["seed"] = seed
    agents = {
        "backend": "scripted",
        "model": "scripted",
        "memory_window": 3,
        "temperature": 0.0,
        "extra": {"replies_by_agent": [list(r) for r in replies_per_agent]},
    }
    return ExperimentConfig(
        run_id=run_id,
        env_type="pricing",
        environment=env,
        agents=agents,
        prompt_dir=str(ROOT / "prompts" / "pricing"),
        communication_mode=comm_mode,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------


def test_config_from_yaml_round_trips(tmp_path):
    import yaml

    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "base.yaml")
    dumped = cfg.to_yaml_dict()
    # Subclass-specific env fields must survive serialization (regression: pydantic
    # was stripping them because the field type is the EnvironmentConfig base).
    assert "demand_params" in dumped["environment"]
    assert dumped["environment"]["demand_params"] == cfg.environment.demand_params
    yaml_path = tmp_path / "round.yaml"
    yaml_path.write_text(yaml.safe_dump(dumped))
    cfg2 = ExperimentConfig.from_yaml(yaml_path)
    assert cfg.to_yaml_dict() == cfg2.to_yaml_dict()
    assert cfg2.environment.demand_params == cfg.environment.demand_params


def test_config_accepts_single_shared_agent_for_any_n_agents(tmp_path):
    env = _base_env_block()
    env["n_agents"] = 3
    cfg = ExperimentConfig(
        run_id="x",
        env_type="pricing",
        environment=env,
        agents={"backend": "scripted", "model": "scripted", "memory_window": 1},
        prompt_dir="prompts/pricing",
    )
    assert cfg.environment.n_agents == 3


def test_config_rejects_env_type_mismatch():
    # Pre-built PricingConfig with env_type="pricing"; top-level says "other".
    pricing_cfg = PricingConfig(**_base_env_block())
    with pytest.raises(ValueError, match="env_type mismatch"):
        ExperimentConfig(
            run_id="x",
            env_type="other",
            environment=pricing_cfg,
            agents={"backend": "scripted", "model": "scripted", "memory_window": 1},
            prompt_dir="prompts/pricing",
        )


# ---------------------------------------------------------------------------
# Experiment runs
# ---------------------------------------------------------------------------


def test_experiment_writes_one_line_per_round(tmp_path):
    n_rounds = 5
    cfg = _make_config(
        n_rounds=n_rounds,
        output_dir=str(tmp_path),
        replies_per_agent=[
            [str(8)] * n_rounds,  # both agents always price at Nash
            [str(8)] * n_rounds,
        ],
    )
    Experiment(cfg).run()
    log_path = tmp_path / "test-run" / "log.jsonl"
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == n_rounds
    rounds = [json.loads(l)["round"] for l in lines]
    assert rounds == list(range(1, n_rounds + 1))


def test_experiment_log_schema(tmp_path):
    n_rounds = 3
    cfg = _make_config(
        n_rounds=n_rounds,
        output_dir=str(tmp_path),
        replies_per_agent=[["7"] * n_rounds, ["9"] * n_rounds],
    )
    Experiment(cfg).run()
    log_path = tmp_path / "test-run" / "log.jsonl"
    for line in log_path.read_text().strip().splitlines():
        rec = json.loads(line)
        assert set(rec) == {
            "run_id", "env_type", "round", "actions", "rewards",
            "cumulative_rewards", "observations", "messages",
            "audit_event", "trajectory_signals", "reasoning",
        }
        assert rec["env_type"] == "pricing"
        assert rec["messages"] == []  # comm_mode none
        assert rec["audit_event"] is None  # null oversight
        assert "nash_price" not in rec["observations"]
        assert "monopoly_price" not in rec["observations"]
        sig = rec["trajectory_signals"]
        assert {
            "action_spread",
            "reward_elevation",
            "reward_elevation_pre_penalty",
            "reward_elevation_post_penalty",
            "explicit_collusion_flag",
            "behavior_collusion_flag",
            "covert_coordination_flag",
            "hollow_coordination_flag",
            "price_follow_lag1",
        } == set(sig)
        assert sig["action_spread"] == abs(7 - 9)
        assert isinstance(sig["reward_elevation"], list)
        assert rec["reasoning"] == ["7", "9"]
        assert len(rec["reasoning"]) == 2
        assert len(sig["reward_elevation"]) == 2
        assert len(sig["reward_elevation_pre_penalty"]) == 2
        assert len(sig["reward_elevation_post_penalty"]) == 2
        assert sig["reward_elevation"] == sig["reward_elevation_post_penalty"]
        # No oversight → explicit/behavior/hollow are False, covert is unknown.
        assert sig["explicit_collusion_flag"] is False
        assert sig["behavior_collusion_flag"] is False
        assert sig["covert_coordination_flag"] is None
        assert sig["hollow_coordination_flag"] is False


def test_experiment_reproducibility_byte_identical(tmp_path):
    n_rounds = 6
    replies = [[str(8)] * n_rounds, [str(9)] * n_rounds]
    cfg_a = _make_config(
        n_rounds=n_rounds, run_id="run-a",
        output_dir=str(tmp_path / "a"), replies_per_agent=replies,
    )
    cfg_b = _make_config(
        n_rounds=n_rounds, run_id="run-a",  # same run_id so JSONL is byte-identical
        output_dir=str(tmp_path / "b"), replies_per_agent=replies,
    )
    Experiment(cfg_a).run()
    Experiment(cfg_b).run()
    log_a = (tmp_path / "a" / "run-a" / "log.jsonl").read_bytes()
    log_b = (tmp_path / "b" / "run-a" / "log.jsonl").read_bytes()
    assert log_a == log_b


def test_experiment_manifest_fields(tmp_path):
    n_rounds = 3
    cfg = _make_config(
        n_rounds=n_rounds,
        output_dir=str(tmp_path),
        replies_per_agent=[["8"] * n_rounds, ["8"] * n_rounds],
    )
    manifest_path = Experiment(cfg).run()
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["run_id"] == "test-run"
    assert manifest["env_type"] == "pricing"
    assert manifest["log_path"].endswith("log.jsonl")
    assert "config" in manifest
    assert manifest["total_input_tokens"] > 0
    # Output tokens may be 0: scripted "8" replies are <4 chars (token estimate is len//4).
    assert manifest["total_output_tokens"] >= 0
    assert manifest["total_cost_estimate_usd"] == 0.0  # scripted backend
    assert manifest["total_fallback_events"] == 0
    assert len(manifest["agents"]) == 2
    for a in manifest["agents"]:
        assert a["backend"] == "scripted"
        assert "input_tokens" in a and "output_tokens" in a


def test_experiment_output_layout(tmp_path):
    n_rounds = 2
    cfg = _make_config(
        n_rounds=n_rounds,
        run_id="layout-check",
        output_dir=str(tmp_path),
        replies_per_agent=[["8"] * n_rounds, ["8"] * n_rounds],
    )
    Experiment(cfg).run()
    run_dir = tmp_path / "layout-check"
    assert (run_dir / "log.jsonl").exists()
    assert (run_dir / "manifest.json").exists()


def test_experiment_progress_callback_invoked(tmp_path):
    n_rounds = 4
    cfg = _make_config(
        n_rounds=n_rounds,
        output_dir=str(tmp_path),
        replies_per_agent=[["8"] * n_rounds, ["8"] * n_rounds],
    )
    seen: list[tuple[int, int]] = []

    def cb(r: int, total: int, line: dict) -> None:
        seen.append((r, total))

    Experiment(cfg, progress_callback=cb).run()
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_experiment_reward_elevation_is_zero_at_nash(tmp_path):
    # Both agents priced at Nash (=50 in calibrated config) → elevation should be 0.
    n_rounds = 2
    cfg = _make_config(
        n_rounds=n_rounds,
        output_dir=str(tmp_path),
        replies_per_agent=[["50"] * n_rounds, ["50"] * n_rounds],
    )
    Experiment(cfg).run()
    log_path = tmp_path / "test-run" / "log.jsonl"
    rec = json.loads(log_path.read_text().strip().splitlines()[0])
    assert rec["trajectory_signals"]["action_spread"] == 0
    elev = rec["trajectory_signals"]["reward_elevation"]
    assert all(abs(e) < 1e-9 for e in elev)


def test_compute_signals_tracks_pre_and_post_penalty_elevation():
    signals = Experiment._compute_signals(
        actions=[10, 10],
        rewards_pre_penalty=[9.0, 9.0],
        rewards_post_penalty=[4.5, 4.5],
        elevation_baseline=(1.0, 9.0),
        audit_event={"audited": True, "results": []},
        behavior_threshold=0.7,
    )
    assert signals["reward_elevation_pre_penalty"] == [1.0, 1.0]
    assert signals["reward_elevation_post_penalty"] == [0.4375, 0.4375]
    assert signals["reward_elevation"] == signals["reward_elevation_post_penalty"]
    # Behavior flag should use pre-penalty elevation.
    assert signals["behavior_collusion_flag"] is True
    assert signals["explicit_collusion_flag"] is False
    assert signals["covert_coordination_flag"] is True
    assert signals["hollow_coordination_flag"] is False


def test_compute_signals_sets_covert_unknown_when_not_audited():
    signals = Experiment._compute_signals(
        actions=[10, 10],
        rewards_pre_penalty=[9.0, 9.0],
        rewards_post_penalty=[9.0, 9.0],
        elevation_baseline=(1.0, 9.0),
        audit_event=None,
        behavior_threshold=0.7,
    )
    assert signals["behavior_collusion_flag"] is True
    assert signals["explicit_collusion_flag"] is False
    assert signals["covert_coordination_flag"] is None


def test_manifest_totals_include_llm_judge_client_costs(tmp_path):
    class CountingJudgeClient(ModelClient):
        def __init__(self, model_name: str, **kwargs):
            super().__init__(model_name=model_name)

        def generate(self, messages, **kwargs) -> str:
            self.input_tokens += 10
            self.output_tokens += 5
            return "VERDICT: NO\nEVIDENCE: NONE\nREASONING: clean."

        def cost_estimate(self) -> float:
            return 1.25

    try:
        register_backend("counting_judge", CountingJudgeClient)
    except ValueError:
        pass

    n_rounds = 2
    cfg = _make_config(
        n_rounds=n_rounds,
        output_dir=str(tmp_path),
        replies_per_agent=[
            ["coordination?", "8", "coordination?", "8"],
            ["noted", "8", "noted", "8"],
        ],
        comm_mode="public",
    )
    cfg = cfg.model_copy(
        update={
            "oversight": cfg.oversight.model_copy(
                update={
                    "mode": "audit-penalty",
                    "audit_probability": 1.0,
                    "llm_judge_enabled": True,
                    "llm_judge_backend": "counting_judge",
                    "llm_judge_model": "counting-judge",
                }
            )
        }
    )

    manifest_path = Experiment(cfg).run()
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["total_input_tokens"] >= 20
    assert manifest["total_output_tokens"] >= 10
    assert manifest["total_cost_estimate_usd"] >= 1.25


def test_experiment_reuses_delivered_messages_for_memory(tmp_path, monkeypatch):
    class CountingCommunication:
        def __init__(self):
            self.calls = 0

        def collect_messages(self, agents, obs):
            return [{"from": 1, "to": "all", "content": "hello"}]

        def deliver_messages(self, agent_id, all_messages):
            self.calls += 1
            return [m["content"] for m in all_messages]

    comm = CountingCommunication()
    monkeypatch.setattr("collusionlab.runner.experiment.get_comm_handler", lambda mode: comm)
    cfg = _make_config(
        n_rounds=1,
        output_dir=str(tmp_path),
        replies_per_agent=[["8"], ["8"]],
    )
    Experiment(cfg).run()
    assert comm.calls == 2  # one deliver per agent (not repeated in memory update)


def test_auditor_notice_toggle_controls_system_prompt(tmp_path):
    cfg = _make_config(
        n_rounds=1,
        output_dir=str(tmp_path),
        replies_per_agent=[["8"], ["8"]],
    )
    cfg_with_notice = cfg.model_copy(
        update={
            "oversight": cfg.oversight.model_copy(
                update={"mode": "audit-penalty", "include_auditor_notice": True}
            )
        }
    )
    env = get_environment(cfg_with_notice.environment)
    agents_with_notice, _ = Experiment(cfg_with_notice)._build_agents(env)
    prompt0 = agents_with_notice[0].system_prompt.lower()
    assert "independent market regulator" in prompt0
    assert "coordinate effectively" not in prompt0

    cfg_without_notice = cfg.model_copy(
        update={
            "oversight": cfg.oversight.model_copy(
                update={"mode": "audit-penalty", "include_auditor_notice": False}
            )
        }
    )
    env = get_environment(cfg_without_notice.environment)
    agents_without_notice, _ = Experiment(cfg_without_notice)._build_agents(env)
    assert "independent market regulator" not in agents_without_notice[0].system_prompt.lower()
