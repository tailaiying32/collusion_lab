"""Communication handlers.

Each handler decides which agents speak each round and who receives each
message.  The runner loop calls ``collect_messages`` then ``deliver_messages``
per agent — swapping the registered handler is the only change needed to
switch communication topology.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collusionlab.agents.llm_agent import LLMAgent


class CommunicationHandler(ABC):
    """Decides which agents speak each round and who receives each message.

    ``collect_messages`` returns a list of ``{from, to, content}`` dicts.
    ``to`` is either ``"all"`` (public) or an ``int`` agent_id (private).
    The runner writes this list to JSONL under ``messages`` and feeds
    per-recipient slices back to ``decide_action``.
    """

    @abstractmethod
    def collect_messages(
        self, agents: list["LLMAgent"], obs: dict
    ) -> list[dict]: ...

    @abstractmethod
    def deliver_messages(
        self, agent_id: int, all_messages: list[dict]
    ) -> list[str]: ...


class NoCommunication(CommunicationHandler):
    """Comm mode ``none``: no agent speaks, no one receives anything."""

    def collect_messages(self, agents: list["LLMAgent"], obs: dict) -> list[dict]:
        return []

    def deliver_messages(self, agent_id: int, all_messages: list[dict]) -> list[str]:
        return []


class PublicCommunication(CommunicationHandler):
    """Comm mode ``public``: every agent speaks, every agent sees all messages."""

    def collect_messages(self, agents: list["LLMAgent"], obs: dict) -> list[dict]:
        messages: list[dict] = []
        for agent in agents:
            content = agent.compose_message(obs)
            if content is not None:
                messages.append({
                    "from": agent.agent_id,
                    "to": "all",
                    "content": content,
                })
        return messages

    def deliver_messages(self, agent_id: int, all_messages: list[dict]) -> list[str]:
        return [_format_delivered_message(agent_id, m) for m in all_messages]


class PrivateCommunication(CommunicationHandler):
    """Comm mode ``private``: each agent sees only messages addressed to it.

    With 2 agents this is functionally identical to public (each agent's
    message is addressed to the single rival).  The distinction activates
    at 3+ agents where each agent broadcasts one message per recipient.
    """

    def collect_messages(self, agents: list["LLMAgent"], obs: dict) -> list[dict]:
        messages: list[dict] = []
        for agent in agents:
            content = agent.compose_message(obs)
            if content is None:
                continue
            for other in agents:
                if other.agent_id != agent.agent_id:
                    messages.append({
                        "from": agent.agent_id,
                        "to": other.agent_id,
                        "content": content,
                    })
        return messages

    def deliver_messages(self, agent_id: int, all_messages: list[dict]) -> list[str]:
        return [
            _format_delivered_message(agent_id, m)
            for m in all_messages
            if m["to"] == agent_id or m["to"] == "all"
        ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[CommunicationHandler]] = {
    "none": NoCommunication,
    "public": PublicCommunication,
    "private": PrivateCommunication,
}


def register_comm_handler(mode: str, cls: type[CommunicationHandler]) -> None:
    if mode in _REGISTRY:
        raise ValueError(f"comm mode {mode!r} already registered")
    _REGISTRY[mode] = cls


def get_comm_handler(mode: str) -> CommunicationHandler:
    if mode not in _REGISTRY:
        raise KeyError(
            f"unknown comm mode {mode!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[mode]()


def registered_comm_modes() -> list[str]:
    return sorted(_REGISTRY)


def _format_delivered_message(agent_id: int, message: dict) -> str:
    content = message.get("content", "")
    sender = message.get("from")
    if sender == agent_id:
        return f"You said: {content}"
    return f"Agent {sender} said: {content}"
