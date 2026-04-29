"""Pricing-specific per-round trajectory signals.

Appended to the base ``trajectory_signals`` dict by the runner via
``env.compute_extra_signals()``. Keeps pricing-specific signal logic
co-located with the environment that defines their meaning.
"""

from __future__ import annotations

from typing import Any


def compute_pricing_signals(
    actions: list,
    prev_actions: list | None,
) -> dict[str, Any]:
    """Compute pricing-specific signals for one round.

    Parameters
    ----------
    actions:
        Current round's actions (prices) for all agents.
    prev_actions:
        Previous round's actions, or ``None`` for round 1.

    Returns
    -------
    Dict of signal name -> value, merged into ``trajectory_signals``.
    """
    signals: dict[str, Any] = {}

    if prev_actions is not None and len(actions) >= 2 and len(prev_actions) >= 2:
        follow_count = 0
        pair_count = 0
        for i in range(len(actions)):
            for j in range(len(prev_actions)):
                if i == j:
                    continue
                pair_count += 1
                if abs(actions[i] - prev_actions[j]) <= 1:
                    follow_count += 1
        signals["price_follow_lag1"] = follow_count / pair_count if pair_count > 0 else 0.0
    else:
        signals["price_follow_lag1"] = None

    return signals
