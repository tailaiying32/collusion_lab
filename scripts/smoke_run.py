"""End-to-end smoke test against the real OpenAI API.

Runs a short pricing game (10 rounds, 2 agents, no communication, no oversight)
to validate that PricingGame + LLMAgent + OpenAIModelClient compose correctly
before the Phase 3 runner is built. Prints per-round actions and the final
token + cost totals.

Usage:
    mamba run -n collusion_lab python scripts/smoke_run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from collusionlab.agents.llm_agent import LLMAgent
from collusionlab.agents.memory import AgentMemory
from collusionlab.agents.model_client import get_model_client
from collusionlab.environments.pricing import PricingConfig, PricingGame


N_ROUNDS = 10
MODEL = "gpt-4o-mini"


def main() -> None:
    load_dotenv(ROOT / ".env")

    with (ROOT / "configs" / "base.yaml").open() as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["environment"]
    env_cfg.pop("_calibration_note", None)
    env_cfg["n_rounds"] = N_ROUNDS
    game = PricingGame(PricingConfig(**env_cfg))
    obs = game.reset(seed=env_cfg.get("seed", 0))

    prompt_dir = ROOT / "prompts" / "pricing"
    system_template = (prompt_dir / "system.txt").read_text(encoding="utf-8")
    action_template = (prompt_dir / "action_turn.txt").read_text(encoding="utf-8")
    message_template = (prompt_dir / "message_turn.txt").read_text(encoding="utf-8")

    # Pricing-specific system-prompt vars. In Phase 3 this becomes
    # env.system_prompt_vars(agent_id) on the GameEnvironment ABC.
    def system_prompt_for(agent_id: int) -> str:
        return system_template.format(
            agent_id=agent_id,
            n_agents=env_cfg["n_agents"],
            n_rounds=N_ROUNDS,
            price_min=env_cfg["price_min"],
            price_max=env_cfg["price_max"],
            cost=env_cfg["demand_params"]["c"],
            auditor_notice="",
        )

    # One ModelClient per agent so token totals are tracked separately.
    agents: list[LLMAgent] = []
    for i in range(env_cfg["n_agents"]):
        client = get_model_client("openai", model_name=MODEL, temperature=0.2)
        agents.append(
            LLMAgent(
                agent_id=i,
                env=game,
                model_client=client,
                memory=AgentMemory(window_size=5),
                system_prompt=system_prompt_for(i),
                action_turn_template=action_template.replace("{strategic_guidance}", ""),
                message_turn_template=message_template.replace("{strategic_guidance}", ""),
                comm_mode="none",
                n_rounds=N_ROUNDS,
            )
        )

    print(f"Running {N_ROUNDS} rounds, {env_cfg['n_agents']} agents, model={MODEL}")
    print(f"Nash price = {game._nash_price}, monopoly price = {game._monopoly_price}")
    print("-" * 60)

    for r in range(1, N_ROUNDS + 1):
        actions = [a.decide_action(obs, messages_received=[]) for a in agents]
        rewards, obs, done = game.step(actions)
        for a, own_action, own_reward in zip(agents, actions, rewards):
            a.memory.update({
                "round": r,
                "own_action": own_action,
                "all_actions": list(actions),
                "own_reward": own_reward,
                "penalty_applied": False,
                "messages_received": [],
                "message_sent": None,
                "own_reasoning": a.last_reasoning,
            })
        print(f"round {r:2d}: actions={actions}  rewards={[round(x, 2) for x in rewards]}")
        if done:
            break

    print("-" * 60)
    total_in = sum(a.model_client.input_tokens for a in agents)
    total_out = sum(a.model_client.output_tokens for a in agents)
    total_cost = sum(a.model_client.cost_estimate() for a in agents)
    print(f"tokens: input={total_in}  output={total_out}")
    print(f"estimated cost: ${total_cost:.5f}")
    fallbacks = sum(len(a.fallback_events) for a in agents)
    if fallbacks:
        print(f"WARNING: {fallbacks} fallback event(s) — agents failed to parse 3x")


if __name__ == "__main__":
    main()
