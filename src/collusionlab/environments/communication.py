"""Communication handlers.

Phase 3 ships `NoCommunication` only. Phase 4 adds `PublicCommunication` and
`PrivateCommunication`; the runner loop does not change — it only swaps which
handler is registered under which mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collusionlab.agents.llm_agent import LLMAgent


class CommunicationHandler(ABC):
    """Decides which agents speak each round and who receives each message.

    `collect_messages` returns a list of `{from, to, content}` dicts. `to` is either
    "all" (public) or an agent_id (private/none). The runner writes this list to
    JSONL under `messages` and feeds per-recipient slices back to `decide_action`.
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
    """Comm mode `none`: no agent speaks, no one receives anything."""

    def collect_messages(self, agents: list["LLMAgent"], obs: dict) -> list[dict]:
        return []

    def deliver_messages(self, agent_id: int, all_messages: list[dict]) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, type[CommunicationHandler]] = {"none": NoCommunication}


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
