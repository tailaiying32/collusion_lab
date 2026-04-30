"""Bounded sliding-window memory for an LLM agent.

Stores per-round records with a fixed env-agnostic schema (see `update()` docstring)
and serializes them into compact text for prompt injection.

Generic labels (`action`, `reward`) are used in the rendered context. The system
prompt is responsible for telling the model what those mean in the current
environment, which keeps memory rendering fully env-agnostic.
"""

from __future__ import annotations

from collections import deque
from typing import Any

MAX_REASONING_CHARS = 240


REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "round",
        "own_action",
        "all_actions",
        "own_quantity",
        "all_quantities",
        "own_reward",
        "penalty_applied",
        "auditor_feedback",
        "messages_received",
        "message_sent",
        "own_reasoning",
    }
)


class AgentMemory:
    def __init__(self, window_size: int) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self._buf: deque[dict] = deque(maxlen=window_size)

    def update(self, round_data: dict) -> None:
        """Append a round and drop the oldest if over window.

        `round_data` keys (exact, env-agnostic):
            round: int
            own_action: any
            all_actions: list  -- every agent's action that round, in agent_id order
            own_quantity: float  -- own demand share (numerator / full denom incl. outside option)
            all_quantities: list[float]  -- all firms' demand shares, same order as all_actions
            own_reward: float  -- own reward only; rivals' rewards are private
            penalty_applied: bool
            auditor_feedback: str
            messages_received: list[str]
            message_sent: str | None
            own_reasoning: str | None  -- prior action-turn text (private to this agent)
        """
        keys = set(round_data.keys())
        missing = REQUIRED_KEYS - keys
        extra = keys - REQUIRED_KEYS
        if missing or extra:
            raise ValueError(
                f"round_data schema mismatch (missing={sorted(missing)}, "
                f"extra={sorted(extra)})"
            )
        cleaned = dict(round_data)  # defensive copy
        cleaned["own_reasoning"] = _clip_reasoning(cleaned.get("own_reasoning"))
        self._buf.append(cleaned)

    def __len__(self) -> int:
        return len(self._buf)

    def records(self) -> list[dict]:
        return list(self._buf)

    def to_prompt_context(self) -> str:
        """Render memory as compact human-readable text for prompt injection.

        Returns the empty string when memory is empty (round 1, before any step).
        Format per round:

            Round {n}: your action={a}, all actions={...}, your market share={s:.1f}%, your reward={r:.4f}
              you said: "..."   (omitted if None)
              you received: "..." | "..."   (omitted if empty)
              you reasoned: "..."   (omitted if None)
        """
        if not self._buf:
            return ""
        lines: list[str] = []
        for r in self._buf:
            penalty_note = "  ← penalty applied this round" if r.get("penalty_applied") else ""
            total_q = sum(r["all_quantities"])
            active_share = r["own_quantity"] / total_q if total_q > 0 else 0.0
            head = (
                f"Round {r['round']}: your action={_fmt(r['own_action'])}, "
                f"all actions={_fmt_list(r['all_actions'])}, "
                f"your market share={_fmt_pct(active_share)}, "
                f"your reward={_fmt_reward(r['own_reward'])}{penalty_note}"
            )
            lines.append(head)
            if r["message_sent"] is not None:
                lines.append(f'  you said: "{r["message_sent"]}"')
            if r["messages_received"]:
                joined = " | ".join(f'"{m}"' for m in r["messages_received"])
                lines.append(f"  you received: {joined}")
            lines.append(f"  {r['auditor_feedback']}")
            own_reasoning = r.get("own_reasoning")
            if own_reasoning:
                lines.append(f'  you reasoned: "{own_reasoning}"')
        return "\n".join(lines)


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _fmt_list(xs: list) -> str:
    return "[" + ", ".join(_fmt(x) for x in xs) + "]"


def _fmt_reward(x: float) -> str:
    return f"{x:.4f}"


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _clip_reasoning(text: Any) -> str | None:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if len(s) <= MAX_REASONING_CHARS:
        return s
    return s[: MAX_REASONING_CHARS - 15].rstrip() + "... [truncated]"
