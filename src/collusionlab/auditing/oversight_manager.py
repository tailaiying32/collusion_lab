"""Oversight manager — auditor orchestration, probability rolling, penalties.

Holds a list of ``Auditor`` instances.  Each round, rolls a seeded probability
die to decide whether an audit occurs.  If it fires, runs all auditors and
applies a reward penalty when any auditor flags the round.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from collusionlab.auditing.base import Auditor
from collusionlab.auditing.behavior_auditor import BehaviorAuditor
from collusionlab.auditing.risk_fusion_policy import RiskFusionPolicy
from collusionlab.auditing.temporal_auditor import TemporalAuditor
from collusionlab.auditing.transcript_auditor import TranscriptAuditor

if TYPE_CHECKING:
    from collusionlab.environments.base import GameEnvironment
    from collusionlab.runner.config import OversightConfig


class OversightManager:
    """Orchestrates per-round auditing and penalty application.

    Parameters
    ----------
    auditors:
        List of ``Auditor`` instances to run when an audit fires.
    audit_probability:
        Per-round probability that an audit occurs (0 = never, 1 = always).
    penalty_factor:
        Multiplicative factor applied to rewards when any auditor flags.
        1.0 = no penalty, 0.5 = half profit confiscated.
    seed:
        Seed for the audit probability RNG (reproducibility).
    """

    def __init__(
        self,
        auditors: list[Auditor] | None = None,
        audit_probability: float = 0.0,
        penalty_factor: float = 1.0,
        fusion_policy: RiskFusionPolicy | None = None,
        seed: int = 0,
    ) -> None:
        self.auditors = auditors or []
        self.audit_probability = audit_probability
        self.penalty_factor = penalty_factor
        self.fusion_policy = fusion_policy or RiskFusionPolicy()
        self._rng = random.Random(seed)

    @classmethod
    def from_config(
        cls,
        config: "OversightConfig",
        seed: int,
        env: "GameEnvironment",
    ) -> "OversightManager":
        """Build an ``OversightManager`` from config + environment context."""
        if config.mode == "none":
            return cls(
                auditors=[],
                audit_probability=0.0,
                penalty_factor=1.0,
                seed=seed,
            )

        auditors: list[Auditor] = []

        auditors.append(
            TranscriptAuditor(
                keywords=list(config.keywords),
                semantic_enabled=config.semantic_enabled,
                semantic_threshold=config.semantic_threshold,
            )
        )

        baseline = env.reward_elevation_baseline()
        baseline_reward = baseline[0] if baseline else 0.0
        ceiling_reward = baseline[1] if baseline else 1.0
        auditors.append(BehaviorAuditor(
            window=config.behavior_window,
            threshold=config.behavior_threshold,
            baseline_reward=baseline_reward,
            ceiling_reward=ceiling_reward,
            convergence_threshold=config.convergence_threshold,
        ))
        auditors.append(TemporalAuditor())

        return cls(
            auditors=auditors,
            audit_probability=config.audit_probability,
            penalty_factor=config.penalty_factor,
            fusion_policy=RiskFusionPolicy(
                weights=config.risk_weights,
                threshold=config.risk_threshold,
                behavior_gate_min=config.behavior_gate_min,
            ),
            seed=seed,
        )

    def check(self, round_log: dict, history: list[dict]) -> dict | None:
        """Run auditors for this round if the probability die fires.

        Returns an audit event dict for JSONL logging, or ``None`` if no
        audit occurred this round.
        """
        if not self.auditors or self.audit_probability <= 0:
            return None

        if self._rng.random() > self.audit_probability:
            return None

        results = [auditor.audit(round_log, history) for auditor in self.auditors]
        results = [r for r in results if r is not None]
        policy_decision = self.fusion_policy.evaluate(results)
        detected_flag = any(r.get("flagged", False) for r in results)
        enforcement_flag = policy_decision.get("flagged", False)

        return {
            "audited": True,
            "flagged": detected_flag,
            "penalty_applied": enforcement_flag,
            "results": results,
            "policy_decision": policy_decision,
        }

    def apply_penalty(
        self, rewards: list[float], event: dict | None
    ) -> list[float]:
        """Apply penalty to rewards if the audit event is flagged."""
        if event is None or not event.get("penalty_applied", False):
            return list(rewards)
        return [r * self.penalty_factor for r in rewards]
