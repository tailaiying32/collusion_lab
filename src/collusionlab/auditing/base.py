"""Auditor abstract base class.

Each auditor inspects a single round's log and returns a result dict if it
detects something noteworthy, or ``None`` if the round is clean.  Auditors
may be stateful (e.g. ``BehaviorAuditor`` maintains a rolling window).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Auditor(ABC):
    """Base class for all auditors.

    Returns
    -------
    dict | None
        A result dict with at minimum ``{"auditor": str, "flagged": bool}``.
        Extra keys under ``"details"`` are auditor-specific.  ``None`` means
        the round was not inspected or nothing noteworthy was found.
    """

    @abstractmethod
    def audit(
        self, round_log: dict, history: list[dict] | None = None
    ) -> dict | None: ...
