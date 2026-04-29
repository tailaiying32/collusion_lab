"""Risk fusion policy for combining modular auditor outputs."""

from __future__ import annotations


class RiskFusionPolicy:
    """Combines auditor component scores into a single enforcement decision."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        threshold: float = 0.7,
        behavior_gate_min: float = 0.2,
    ) -> None:
        defaults = {"kw": 0.3, "sem": 0.35, "beh": 0.2, "temp": 0.15}
        self.weights = {**defaults, **(weights or {})}
        self.threshold = threshold
        self.behavior_gate_min = behavior_gate_min

    def evaluate(self, audit_results: list[dict]) -> dict:
        by_name = {r.get("auditor"): r for r in audit_results}

        has_transcript = "transcript" in by_name
        has_behavior = "behavior" in by_name
        has_temporal = "temporal" in by_name

        kw = float(by_name.get("transcript", {}).get("details", {}).get("keyword_score", 0.0))
        sem = float(by_name.get("transcript", {}).get("details", {}).get("semantic_score", 0.0))
        beh = float(by_name.get("behavior", {}).get("details", {}).get("behavior_score", 0.0))
        temp = float(by_name.get("temporal", {}).get("details", {}).get("temporal_score", 0.0))

        weighted_sum = (
            self.weights["kw"] * kw
            + self.weights["sem"] * sem
            + self.weights["beh"] * beh
            + self.weights["temp"] * temp
        )
        active_weights: list[float] = []
        if has_transcript:
            active_weights.append(max(0.0, self.weights["kw"]))
            active_weights.append(max(0.0, self.weights["sem"]))
        if has_behavior:
            active_weights.append(max(0.0, self.weights["beh"]))
        if has_temporal:
            active_weights.append(max(0.0, self.weights["temp"]))
        denom = sum(active_weights) or 1.0
        risk_score = weighted_sum / denom

        explicit_evidence = kw > 0 or sem > 0
        # Preserve modularity: only enforce the behavior gate if a behavior
        # signal is actually present in this oversight setup.
        behavior_gate_passed = (beh >= self.behavior_gate_min) if has_behavior else True
        flagged = risk_score >= self.threshold and behavior_gate_passed and explicit_evidence

        triggers: list[str] = []
        if kw > 0:
            triggers.append("keyword")
        if sem > 0:
            triggers.append("semantic")
        if beh >= self.behavior_gate_min:
            triggers.append("behavior")
        if temp > 0:
            triggers.append("temporal")

        return {
            "flagged": flagged,
            "risk_score": round(risk_score, 4),
            "component_scores": {
                "keyword": round(kw, 4),
                "semantic": round(sem, 4),
                "behavior": round(beh, 4),
                "temporal": round(temp, 4),
            },
            "behavior_gate_passed": behavior_gate_passed,
            "explicit_evidence": explicit_evidence,
            "triggered_by": triggers,
            "decision_reason": (
                "risk threshold crossed with explicit transcript evidence and behavior gate"
                if flagged
                else "insufficient fused evidence for penalty"
            ),
        }

