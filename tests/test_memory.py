"""Unit tests for AgentMemory rendering and auditor feedback summaries."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from collusionlab.agents.memory import AgentMemory
from collusionlab.runner.experiment import Experiment


def _round_data(**overrides) -> dict:
    base = {
        "round": 1,
        "own_action": 50,
        "all_actions": [50, 50],
        "own_quantity": 0.2,
        "all_quantities": [0.2, 0.2],
        "own_reward": 76.08,
        "penalty_applied": False,
        "auditor_feedback": "Auditor outcome: audited, no violation flagged.",
        "messages_received": [],
        "message_sent": None,
        "own_reasoning": "I will keep the current price.",
        "quarterly_report": None,
    }
    return {**base, **overrides}


def test_memory_schema_requires_auditor_feedback():
    mem = AgentMemory(window_size=3)
    bad = _round_data()
    bad.pop("auditor_feedback")
    with pytest.raises(ValueError, match="schema mismatch"):
        mem.update(bad)


def test_memory_context_includes_auditor_feedback_line():
    mem = AgentMemory(window_size=3)
    mem.update(
        _round_data(
            penalty_applied=True,
            auditor_feedback=(
                "Auditor outcome: FLAGGED, PENALTY APPLIED. Reason: "
                "suspected explicit coordination in monitored communications."
            ),
        )
    )
    ctx = mem.to_prompt_context()
    assert "penalty applied this round" in ctx
    assert "Auditor outcome: FLAGGED, PENALTY APPLIED." in ctx
    assert "Reason: suspected explicit coordination in monitored communications." in ctx


def test_auditor_feedback_not_audited_branch():
    msg = Experiment._build_auditor_feedback(None)
    assert msg == "Auditor outcome: not audited this round."


def test_auditor_feedback_flagged_no_penalty_branch():
    event = {
        "audited": True,
        "flagged": True,
        "penalty_applied": False,
        "results": [{"auditor": "behavior", "flagged": True, "details": {}}],
        "policy_decision": {},
    }
    msg = Experiment._build_auditor_feedback(event)
    assert "FLAGGED, NO PENALTY APPLIED" in msg
    assert "sustained coordinated pricing pattern" in msg


def test_auditor_feedback_is_sanitized_no_raw_evidence_leak():
    event = {
        "audited": True,
        "flagged": True,
        "penalty_applied": True,
        "results": [
            {
                "auditor": "llm_judge",
                "flagged": True,
                "details": {
                    "raw_response": "VERDICT: YES ...",
                    "evidence": "Agent said: set price to 80",
                    "reasoning": "contains explicit coordination",
                },
            },
            {
                "auditor": "transcript",
                "flagged": True,
                "details": {"excerpts": ["set price to 80"]},
            },
        ],
        "policy_decision": {"enforcement_mode": "judge_or_behavior"},
    }
    msg = Experiment._build_auditor_feedback(event)
    assert "FLAGGED, PENALTY APPLIED" in msg
    assert "suspected explicit coordination in monitored communications" in msg
    assert "VERDICT:" not in msg
    assert "set price to 80" not in msg
