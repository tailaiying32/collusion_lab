"""Single-run experiment driver.

Loads an ExperimentConfig, instantiates the environment + agents + comm handler +
oversight manager, and runs the round loop. Writes one JSON object per round to
`{output_dir}/{run_id}/log.jsonl` and a `manifest.json` alongside it. Each round
record includes a per-agent `reasoning` list (private action-turn text, not
visible to other agents or the auditor). Designed so Phase 4 (real comm +
oversight) plugs in via registry without changing this file.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from collusionlab.agents.llm_agent import LLMAgent
from collusionlab.agents.memory import AgentMemory
from collusionlab.agents.model_client import ModelClient, get_model_client
from collusionlab.auditing import OversightManager
from collusionlab.environments.base import GameEnvironment, get_environment
from collusionlab.environments.communication import (
    CommunicationHandler,
    get_comm_handler,
)
from collusionlab.runner.config import AgentConfig, ExperimentConfig


logger = logging.getLogger(__name__)


ProgressCallback = Callable[[int, int, dict], None]


class Experiment:
    """Runs one fully-specified experiment end-to-end."""

    def __init__(
        self,
        config: ExperimentConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.config = config.with_run_id()
        self.progress_callback = progress_callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Path:
        cfg = self.config
        run_dir = Path(cfg.output_dir) / cfg.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "log.jsonl"
        manifest_path = run_dir / "manifest.json"

        env = get_environment(cfg.environment)
        agents, model_clients = self._build_agents(env)
        comm = get_comm_handler(cfg.communication_mode)
        oversight = OversightManager.from_config(
            cfg.oversight, seed=cfg.environment.seed, env=env,
        )

        elevation_baseline = env.reward_elevation_baseline()

        obs = env.reset(cfg.environment.seed)
        cumulative_rewards = [0.0] * env.n_agents
        prev_actions: list | None = None
        history: list[dict] = []
        n_rounds = cfg.environment.n_rounds

        start_time = datetime.now(timezone.utc)
        start_perf = time.perf_counter()

        with log_path.open("w", encoding="utf-8") as log_f:
            for round_idx in range(1, n_rounds + 1):
                # (a) Pre-play communication.
                raw_messages = comm.collect_messages(agents, obs)

                # (b) Each agent decides an action with its delivered messages.
                actions = []
                delivered_by_agent: dict[int, list[str]] = {}
                for agent in agents:
                    delivered = comm.deliver_messages(agent.agent_id, raw_messages)
                    delivered_by_agent[agent.agent_id] = list(delivered)
                    actions.append(agent.decide_action(obs, delivered))
                reasoning = [a.last_reasoning for a in agents]

                # (c) Step the env.
                rewards, next_obs, done = env.step(actions)
                rewards_pre_penalty = list(rewards)

                # (d) Oversight.
                round_log_for_audit = {
                    "round": round_idx,
                    "actions": list(actions),
                    "rewards": list(rewards),
                    "messages": raw_messages,
                }
                audit_event = oversight.check(round_log_for_audit, history)
                rewards = oversight.apply_penalty(rewards, audit_event)
                rewards_post_penalty = list(rewards)
                penalty_applied = bool(
                    audit_event and audit_event.get("penalty_applied")
                )

                # (e) Update each agent's memory with post-penalty rewards.
                for agent in agents:
                    sent = self._message_from(agent.agent_id, raw_messages)
                    agent.memory.update({
                        "round": round_idx,
                        "own_action": actions[agent.agent_id],
                        "all_actions": list(actions),
                        "own_reward": rewards[agent.agent_id],
                        "penalty_applied": penalty_applied,
                        "messages_received": delivered_by_agent[agent.agent_id],
                        "message_sent": sent,
                        "own_reasoning": agent.last_reasoning,
                    })

                cumulative_rewards = [
                    cumulative_rewards[i] + rewards[i] for i in range(env.n_agents)
                ]

                # (f) Trajectory signals.
                signals = self._compute_signals(
                    actions,
                    rewards_pre_penalty,
                    rewards_post_penalty,
                    elevation_baseline,
                    audit_event,
                )
                extra = env.compute_extra_signals(
                    actions, rewards_post_penalty, prev_actions, round_idx,
                )
                signals.update(extra)
                prev_actions = list(actions)

                # (g) Write round log line.
                line = {
                    "run_id": cfg.run_id,
                    "env_type": cfg.env_type,
                    "round": round_idx,
                    "actions": list(actions),
                    "rewards": list(rewards),
                    "cumulative_rewards": list(cumulative_rewards),
                    "observations": next_obs,
                    "messages": raw_messages,
                    "audit_event": audit_event,
                    "trajectory_signals": signals,
                    "reasoning": list(reasoning),
                }
                log_f.write(json.dumps(line, sort_keys=True) + "\n")
                history.append(line)
                if self.progress_callback is not None:
                    self.progress_callback(round_idx, n_rounds, line)

                obs = next_obs
                if done:
                    break

        end_time = datetime.now(timezone.utc)
        elapsed = time.perf_counter() - start_perf
        manifest = self._build_manifest(
            log_path=log_path,
            model_clients=model_clients,
            agents=agents,
            start_time=start_time,
            end_time=end_time,
            elapsed_s=elapsed,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_agents(
        self, env: GameEnvironment
    ) -> tuple[list[LLMAgent], list[ModelClient]]:
        cfg = self.config
        prompt_dir = Path(cfg.prompt_dir)
        system_template = (prompt_dir / "system.txt").read_text(encoding="utf-8")
        action_template = (prompt_dir / "action_turn.txt").read_text(encoding="utf-8")
        message_template = (prompt_dir / "message_turn.txt").read_text(encoding="utf-8")
        auditor_notice_template = (prompt_dir / "auditor_notice.txt").read_text(encoding="utf-8")

        agents: list[LLMAgent] = []
        clients: list[ModelClient] = []
        for agent_id, agent_cfg in enumerate(cfg.agents):
            client = self._build_client(agent_cfg)
            clients.append(client)
            prompt_vars = env.system_prompt_vars(agent_id)
            prompt_vars["auditor_notice"] = (
                "\n\n" + auditor_notice_template.strip()
                if cfg.oversight
                and cfg.oversight.mode == "audit-penalty"
                and cfg.oversight.include_auditor_notice
                else ""
            )
            system_prompt = system_template.format(**prompt_vars)
            agents.append(
                LLMAgent(
                    agent_id=agent_id,
                    env=env,
                    model_client=client,
                    memory=AgentMemory(window_size=agent_cfg.memory_window),
                    system_prompt=system_prompt,
                    action_turn_template=action_template,
                    message_turn_template=message_template,
                    comm_mode=cfg.communication_mode,
                    n_rounds=cfg.environment.n_rounds,
                )
            )
        return agents, clients

    @staticmethod
    def _build_client(agent_cfg: AgentConfig) -> ModelClient:
        kwargs = {"temperature": agent_cfg.temperature, **agent_cfg.extra}
        return get_model_client(
            agent_cfg.backend, model_name=agent_cfg.model, **kwargs
        )

    @staticmethod
    def _message_from(agent_id: int, all_messages: list[dict]) -> str | None:
        for m in all_messages:
            if m.get("from") == agent_id:
                return m.get("content")
        return None

    @staticmethod
    def _compute_signals(
        actions: list,
        rewards_pre_penalty: list[float],
        rewards_post_penalty: list[float],
        elevation_baseline: tuple[float, float] | None,
        audit_event: dict | None,
    ) -> dict:
        try:
            spread = max(actions) - min(actions)
        except TypeError:
            spread = None

        elevation_pre = None
        elevation_post = None
        if elevation_baseline is not None:
            low, high = elevation_baseline
            denom = high - low
            if denom != 0 and math.isfinite(denom):
                elevation_pre = [(r - low) / denom for r in rewards_pre_penalty]
                elevation_post = [(r - low) / denom for r in rewards_post_penalty]

        signals: dict = {}
        if spread is not None:
            signals["action_spread"] = spread
        if elevation_pre is not None:
            signals["reward_elevation_pre_penalty"] = elevation_pre
        if elevation_post is not None:
            # Keep the legacy key for backward compatibility in metrics/UI code.
            signals["reward_elevation"] = elevation_post
            signals["reward_elevation_post_penalty"] = elevation_post

        explicit = False
        behavior = False
        if audit_event is not None and audit_event.get("audited"):
            for result in audit_event.get("results", []):
                if result.get("auditor") == "transcript" and result.get("flagged"):
                    explicit = True
                if result.get("auditor") == "behavior" and result.get("flagged"):
                    behavior = True
            policy = audit_event.get("policy_decision", {})
            if policy.get("flagged") and policy.get("explicit_evidence"):
                explicit = True

        signals["explicit_collusion_flag"] = explicit
        signals["behavior_collusion_flag"] = behavior
        signals["covert_coordination_flag"] = behavior and not explicit
        signals["hollow_coordination_flag"] = explicit and not behavior

        return signals

    def _build_manifest(
        self,
        log_path: Path,
        model_clients: list[ModelClient],
        agents: list[LLMAgent],
        start_time: datetime,
        end_time: datetime,
        elapsed_s: float,
    ) -> dict:
        cfg = self.config
        per_agent = []
        total_in = total_out = 0
        total_cost = 0.0
        total_fallbacks = 0
        for agent_id, (agent, client) in enumerate(zip(agents, model_clients)):
            per_agent.append({
                "agent_id": agent_id,
                "backend": cfg.agents[agent_id].backend,
                "model": client.model_name,
                "input_tokens": client.input_tokens,
                "output_tokens": client.output_tokens,
                "cost_estimate_usd": client.cost_estimate(),
                "fallback_events": list(agent.fallback_events),
            })
            total_in += client.input_tokens
            total_out += client.output_tokens
            total_cost += client.cost_estimate()
            total_fallbacks += len(agent.fallback_events)
        return {
            "run_id": cfg.run_id,
            "env_type": cfg.env_type,
            "config": cfg.to_yaml_dict(),
            "log_path": str(log_path),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "elapsed_seconds": elapsed_s,
            "agents": per_agent,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_cost_estimate_usd": total_cost,
            "total_fallback_events": total_fallbacks,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a single CollusionLab experiment.")
    parser.add_argument("--config", required=True, help="Path to ExperimentConfig YAML.")
    parser.add_argument("--output-dir", help="Override output_dir from the config.")
    parser.add_argument("--run-id", help="Override run_id from the config.")
    parser.add_argument(
        "--log-level", default="INFO", help="Python logging level (default INFO)."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()

    cfg = ExperimentConfig.from_yaml(args.config)
    overrides: dict = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.run_id:
        overrides["run_id"] = args.run_id
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    manifest_path = Experiment(cfg).run()
    print(str(manifest_path))


if __name__ == "__main__":
    main()
