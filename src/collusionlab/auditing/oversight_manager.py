"""Oversight manager.

Phase 3 ships a null implementation that never fires and never penalizes; the
runner writes `audit_event: null` on every line. Phase 4 expands this with real
auditors and probability rolling without the runner loop changing.
"""

from __future__ import annotations


class OversightManager:
    """Null oversight: no auditors, no penalties.

    Constructor accepts arbitrary kwargs so Phase 3 configs can carry the
    Phase-4 oversight fields (audit_probability, penalty_factor, ...) without the
    runner needing two code paths.
    """

    def __init__(self, mode: str = "none", **_: object) -> None:
        self.mode = mode

    def check(self, round_log: dict, history: list[dict]) -> dict | None:
        return None

    def apply_penalty(self, rewards: list[float], event: dict | None) -> list[float]:
        return list(rewards)
