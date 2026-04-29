"""Environment-agnostic LLM agent.

Holds a `ModelClient`, an `AgentMemory`, a pre-rendered system prompt, and per-turn
templates. Action parsing is delegated to the `GameEnvironment`, so this class
contains no environment-specific logic.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from collusionlab.agents.memory import AgentMemory
from collusionlab.agents.model_client import ModelClient
from collusionlab.environments.base import GameEnvironment


logger = logging.getLogger(__name__)


CommMode = Literal["none", "public", "private"]


class LLMAgent:
    def __init__(
        self,
        agent_id: int,
        env: GameEnvironment,
        model_client: ModelClient,
        memory: AgentMemory,
        system_prompt: str,
        action_turn_template: str,
        message_turn_template: str,
        comm_mode: CommMode,
        n_rounds: int,
        max_action_attempts: int = 3,
    ) -> None:
        self.agent_id = agent_id
        self.env = env
        self.model_client = model_client
        self.memory = memory
        self.system_prompt = system_prompt
        self.action_turn_template = action_turn_template
        self.message_turn_template = message_turn_template
        self.comm_mode: CommMode = comm_mode
        self.n_rounds = n_rounds
        self.max_action_attempts = max_action_attempts
        self.fallback_events: list[dict] = []
        self.last_reasoning: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compose_message(self, obs: dict) -> str | None:
        """Pre-play turn. Returns None when comm mode is `none`.

        `obs` is the most recent obs available: the reset obs in round 1, the prior
        step()'s obs thereafter (see Phase 2.7 spec).
        """
        if self.comm_mode == "none":
            return None
        prompt = self.message_turn_template.format(
            round_number=self._current_round_number(obs),
            n_rounds=self.n_rounds,
            memory_context=self._memory_context(),
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.model_client.generate(messages).strip()

    def decide_action(self, obs: dict, messages_received: list[str]) -> Any:
        """Action turn. Returns a parsed action; falls back to env.default_action()
        after max_action_attempts unparseable replies.
        """
        action_desc = self.env.action_space()["description"]
        user_prompt = self.action_turn_template.format(
            round_number=self._current_round_number(obs),
            n_rounds=self.n_rounds,
            memory_context=self._memory_context(),
            recent_messages=self._format_received(messages_received),
            action_space_description=action_desc,
        )
        conversation: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        attempts: list[dict] = []
        for _ in range(self.max_action_attempts):
            text = self.model_client.generate(conversation)
            try:
                action = self.env.parse_action(text)
                self.last_reasoning = text
                return action
            except ValueError as e:
                attempts.append({"raw": text, "error": str(e)})
                conversation.append({"role": "assistant", "content": text})
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response could not be parsed: {e}. "
                            f"Please respond with only {action_desc}."
                        ),
                    }
                )
        # Retry budget exhausted: log and fall back.
        event = {
            "round": self._current_round_number(obs),
            "agent_id": self.agent_id,
            "attempts": attempts,
        }
        self.fallback_events.append(event)
        self.last_reasoning = attempts[-1]["raw"] if attempts else None
        logger.warning(
            "agent %d fell back to default_action after %d failed attempts",
            self.agent_id,
            self.max_action_attempts,
        )
        return self.env.default_action()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _current_round_number(obs: dict) -> int:
        # obs["round"] counts completed rounds. The round we're about to play is
        # one beyond that, 1-indexed for the prompt.
        return int(obs.get("round", 0)) + 1

    def _memory_context(self) -> str:
        ctx = self.memory.to_prompt_context()
        return ctx if ctx else "(none)"

    @staticmethod
    def _format_received(messages_received: list[str]) -> str:
        if not messages_received:
            return "(none)"
        return "\n".join(f"- {m}" for m in messages_received)
