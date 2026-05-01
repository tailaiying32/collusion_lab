"""Phase 2 unit tests for the agent layer."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from collusionlab.agents.llm_agent import LLMAgent
from collusionlab.agents.memory import AgentMemory
from collusionlab.agents.model_client import (
    ModelClient,
    get_model_client,
    registered_backends,
)
from collusionlab.environments.pricing import PricingConfig, PricingGame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calibrated_pricing_game(n_rounds: int = 5) -> PricingGame:
    import yaml

    with (ROOT / "configs" / "base.yaml").open() as f:
        env_cfg = yaml.safe_load(f)["environment"]
    env_cfg.pop("_calibration_note", None)
    env_cfg["n_rounds"] = n_rounds
    cfg = PricingConfig(**env_cfg)
    game = PricingGame(cfg)
    game.reset(seed=0)
    return game


class ScriptedModelClient(ModelClient):
    """Fake ModelClient that returns a pre-scripted sequence of replies."""

    def __init__(self, replies: list[str], model_name: str = "fake") -> None:
        super().__init__(model_name=model_name)
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    def generate(self, messages, **kwargs) -> str:
        self.calls.append(list(messages))
        if not self._replies:
            raise RuntimeError("ScriptedModelClient ran out of replies")
        reply = self._replies.pop(0)
        # Simulate token accounting (rough: 4 chars per token).
        self.input_tokens += sum(len(m["content"]) for m in messages) // 4
        self.output_tokens += len(reply) // 4
        return reply

    def cost_estimate(self) -> float:
        return 0.0


def _read_template(name: str) -> str:
    return (ROOT / "prompts" / "pricing" / f"{name}.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# AgentMemory
# ---------------------------------------------------------------------------


def _round(round_no: int, own_action=5, all_actions=None, own_quantity=0.5,
           all_quantities=None, own_reward=1.0, penalty_applied=False,
           auditor_feedback="", messages_received=None, message_sent=None,
           own_reasoning=None, quarterly_report=None) -> dict:
    _all_actions = all_actions if all_actions is not None else [own_action, own_action]
    return {
        "round": round_no,
        "own_action": own_action,
        "all_actions": _all_actions,
        "own_quantity": own_quantity,
        "all_quantities": all_quantities if all_quantities is not None else [own_quantity] * len(_all_actions),
        "own_reward": own_reward,
        "penalty_applied": penalty_applied,
        "auditor_feedback": auditor_feedback,
        "messages_received": messages_received if messages_received is not None else [],
        "message_sent": message_sent,
        "own_reasoning": own_reasoning,
        "quarterly_report": quarterly_report,
    }


def test_memory_window_truncates():
    mem = AgentMemory(window_size=3)
    for i in range(5):
        mem.update(_round(i + 1))
    assert len(mem) == 3
    rs = mem.records()
    assert [r["round"] for r in rs] == [3, 4, 5]


def test_memory_update_rejects_bad_schema():
    mem = AgentMemory(window_size=3)
    with pytest.raises(ValueError, match="schema mismatch"):
        mem.update({"round": 1})  # missing keys
    with pytest.raises(ValueError, match="schema mismatch"):
        bad = _round(1)
        bad["extra_field"] = 1
        mem.update(bad)


def test_memory_to_prompt_context_uses_generic_labels():
    mem = AgentMemory(window_size=2)
    mem.update(_round(1, own_action=7, all_actions=[7, 8], own_reward=2.5,
                       messages_received=["hi"], message_sent="ok"))
    mem.update(_round(2, own_action=8, all_actions=[8, 8], own_reward=3.1))
    ctx = mem.to_prompt_context()
    assert "Round 1" in ctx and "Round 2" in ctx
    # Generic labels — no env-specific words like "price" or "profit".
    assert "your action" in ctx
    assert "your reward" in ctx
    assert "price" not in ctx.lower()
    assert "profit" not in ctx.lower()
    # Message rendering.
    assert 'you said: "ok"' in ctx
    assert 'you received: "hi"' in ctx


def test_memory_to_prompt_context_includes_own_reasoning():
    mem = AgentMemory(window_size=2)
    mem.update(_round(1, own_reasoning="Hold steady given recent margins."))
    ctx = mem.to_prompt_context()
    assert 'you reasoned: "Hold steady given recent margins."' in ctx


def test_memory_clips_long_own_reasoning():
    mem = AgentMemory(window_size=2)
    long_reasoning = "x" * 500
    mem.update(_round(1, own_reasoning=long_reasoning))
    ctx = mem.to_prompt_context()
    assert "... [truncated]" in ctx
    assert len(mem.records()[0]["own_reasoning"]) <= 240


def test_memory_empty_context_is_empty_string():
    assert AgentMemory(window_size=3).to_prompt_context() == ""


# ---------------------------------------------------------------------------
# LLMAgent
# ---------------------------------------------------------------------------


def _make_agent(client: ModelClient, comm_mode="none", n_rounds=5,
                memory_window=3) -> tuple[LLMAgent, PricingGame]:
    game = _calibrated_pricing_game(n_rounds=n_rounds)
    agent = LLMAgent(
        agent_id=0,
        env=game,
        model_client=client,
        memory=AgentMemory(window_size=memory_window),
        system_prompt="SYSTEM",
        action_turn_template=_read_template("action_turn"),
        message_turn_template=_read_template("message_turn"),
        comm_mode=comm_mode,
        n_rounds=n_rounds,
    )
    return agent, game


def test_decide_action_first_attempt_success():
    client = ScriptedModelClient(["7"])
    agent, game = _make_agent(client)
    obs = game.reset(seed=0)
    action = agent.decide_action(obs, messages_received=[])
    assert action == 7
    assert agent.last_reasoning == "7"
    assert len(client.calls) == 1
    assert agent.fallback_events == []


def test_decide_action_retries_with_error_message_then_succeeds():
    # First reply unparseable, second within range.
    client = ScriptedModelClient(["banana", "9"])
    agent, game = _make_agent(client)
    obs = game.reset(seed=0)
    action = agent.decide_action(obs, messages_received=[])
    assert action == 9
    assert len(client.calls) == 2
    # Second call's conversation should include the error feedback.
    second = client.calls[1]
    assert any("could not be parsed" in m["content"] for m in second)
    assert agent.fallback_events == []
    assert agent.last_reasoning == "9"


def test_decide_action_falls_back_to_default_after_three_failures():
    client = ScriptedModelClient(["banana", "still no", "definitely not"])
    agent, game = _make_agent(client)
    obs = game.reset(seed=0)
    action = agent.decide_action(obs, messages_received=[])
    assert action == game.default_action()  # Nash price from calibration
    assert len(client.calls) == 3
    assert len(agent.fallback_events) == 1
    event = agent.fallback_events[0]
    assert event["agent_id"] == 0
    assert len(event["attempts"]) == 3
    assert all("error" in a for a in event["attempts"])
    assert agent.last_reasoning == "definitely not"


def test_compose_message_returns_none_when_comm_mode_none():
    client = ScriptedModelClient([])  # never called
    agent, game = _make_agent(client, comm_mode="none")
    obs = game.reset(seed=0)
    assert agent.compose_message(obs) is None
    assert client.calls == []


def test_compose_message_uses_reset_obs_in_round_1_then_step_obs():
    client = ScriptedModelClient(["hello", "round 2 chat"])
    agent, game = _make_agent(client, comm_mode="public")
    # Round 1: obs is the reset obs.
    reset_obs = game.reset(seed=0)
    msg1 = agent.compose_message(reset_obs)
    assert msg1 == "hello"
    # The user prompt for round 1 should reference Round 1.
    assert "Round 1" in client.calls[0][1]["content"]
    # Step the game; round-2 compose_message uses the post-step obs.
    _, step_obs, _ = game.step([7, 7])
    msg2 = agent.compose_message(step_obs)
    assert msg2 == "round 2 chat"
    assert "Round 2" in client.calls[1][1]["content"]


def test_decide_action_includes_action_space_description_in_prompt():
    client = ScriptedModelClient(["7"])
    agent, game = _make_agent(client)
    obs = game.reset(seed=0)
    agent.decide_action(obs, messages_received=["price low"])
    user_msg = client.calls[0][1]["content"]
    assert game.action_space()["description"] in user_msg
    # Received messages also rendered.
    assert "price low" in user_msg


def test_prompts_include_window_size_header_text():
    client = ScriptedModelClient(["hello", "7"])
    agent, game = _make_agent(client, comm_mode="public", memory_window=3)
    obs = game.reset(seed=0)
    agent.compose_message(obs)
    agent.decide_action(obs, messages_received=[])
    assert "Last 3 rounds, chronological order:" in client.calls[0][1]["content"]
    assert "Last 3 rounds, chronological order:" in client.calls[1][1]["content"]


# ---------------------------------------------------------------------------
# ModelClient backends (mocked)
# ---------------------------------------------------------------------------


def test_token_accumulation_across_calls():
    client = ScriptedModelClient(["one", "two", "three"])
    for _ in range(3):
        client.generate([{"role": "user", "content": "hi there"}])
    assert client.input_tokens > 0
    assert client.output_tokens > 0


def test_openai_client_instantiates_and_returns_string_with_mock():
    # Patch the SDK constructor and chat.completions.create call.
    with patch("openai.OpenAI") as openai_cls:
        instance = MagicMock()
        openai_cls.return_value = instance
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="hello"))]
        response.usage = MagicMock(prompt_tokens=10, completion_tokens=3)
        instance.chat.completions.create.return_value = response

        from collusionlab.agents.backends.openai_client import OpenAIModelClient

        c = OpenAIModelClient(model_name="gpt-4o-mini", api_key="sk-test")
        out = c.generate([{"role": "user", "content": "hi"}])
        assert out == "hello"
        assert c.input_tokens == 10
        assert c.output_tokens == 3
        # Cost estimate uses the price table.
        assert c.cost_estimate() > 0


def test_openai_payload_params_by_model_family():
    with patch("openai.OpenAI") as openai_cls:
        instance = MagicMock()
        openai_cls.return_value = instance
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="ok"))]
        response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        instance.chat.completions.create.return_value = response

        from collusionlab.agents.backends.openai_client import OpenAIModelClient

        c_legacy = OpenAIModelClient(model_name="gpt-4o-mini", api_key="sk-test")
        c_legacy.generate([{"role": "user", "content": "hi"}], max_tokens=123)
        kwargs_legacy = instance.chat.completions.create.call_args.kwargs
        assert "temperature" in kwargs_legacy
        assert "max_tokens" in kwargs_legacy
        assert "max_completion_tokens" not in kwargs_legacy

        c_reasoning = OpenAIModelClient(model_name="gpt-5-mini", api_key="sk-test")
        c_reasoning.generate([{"role": "user", "content": "hi"}], max_tokens=456)
        kwargs_reasoning = instance.chat.completions.create.call_args.kwargs
        assert "temperature" not in kwargs_reasoning
        assert "max_completion_tokens" in kwargs_reasoning
        assert "max_tokens" not in kwargs_reasoning


def test_anthropic_client_instantiates_and_returns_string_with_mock():
    with patch("anthropic.Anthropic") as anth_cls:
        instance = MagicMock()
        anth_cls.return_value = instance
        block = MagicMock()
        block.type = "text"
        block.text = "world"
        response = MagicMock()
        response.content = [block]
        response.usage = MagicMock(input_tokens=20, output_tokens=5)
        instance.messages.create.return_value = response

        from collusionlab.agents.backends.anthropic_client import AnthropicModelClient

        c = AnthropicModelClient(model_name="claude-haiku-4-5-20251001", api_key="x")
        out = c.generate(
            [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hi"},
            ]
        )
        assert out == "world"
        assert c.input_tokens == 20
        assert c.output_tokens == 5
        # System message was extracted from the messages list.
        kwargs = instance.messages.create.call_args.kwargs
        assert kwargs["system"] == "be helpful"
        assert all(m["role"] != "system" for m in kwargs["messages"])


def test_deepseek_client_instantiates_and_returns_string_with_mock():
    with patch("openai.OpenAI") as openai_cls:
        instance = MagicMock()
        openai_cls.return_value = instance
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="deep reply"))]
        response.usage = MagicMock(prompt_tokens=7, completion_tokens=2)
        instance.chat.completions.create.return_value = response

        from collusionlab.agents.backends.deepseek_client import DeepSeekModelClient

        c = DeepSeekModelClient(model_name="deepseek-v4-flash", api_key="x")
        out = c.generate([{"role": "user", "content": "hi"}])
        assert out == "deep reply"
        assert c.input_tokens == 7
        assert c.output_tokens == 2
        assert c.cost_estimate() > 0


def test_backend_registry_resolves_lazily():
    backends = registered_backends()
    # Trigger lazy import path.
    with patch("openai.OpenAI"):
        client = get_model_client("openai", model_name="gpt-4o-mini", api_key="sk-test")
    assert client.model_name == "gpt-4o-mini"
    assert "openai" in registered_backends()
    with patch("openai.OpenAI"):
        deep_client = get_model_client("deepseek", model_name="deepseek-v4-flash", api_key="x")
    assert deep_client.model_name == "deepseek-v4-flash"
    assert "deepseek" in registered_backends()
