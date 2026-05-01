"""LLM-as-judge transcript auditor.

Calls a configurable ModelClient to evaluate whether any message in a round
constitutes explicit price coordination. Returns a structured result with
verdict, evidence quote, and one-sentence reasoning. Replaces TranscriptAuditor
+ RiskFusionPolicy as the primary explicit-collusion detector when enabled.

Conservative on failure: malformed responses or model errors return
``flagged=False`` so a single bad call cannot create a spurious penalty.

Deliberately surface-level: this auditor is for same-round explicit language
only. Cross-round code or message-action pattern detection belongs in offline
research analysis, not in the in-game enforcement path.
"""

from __future__ import annotations

import re

from collusionlab.agents.model_client import ModelClient
from collusionlab.auditing.base import Auditor


_VERDICT_RE = re.compile(r"^VERDICT:\s*(YES|NO)\b", re.IGNORECASE | re.MULTILINE)
_EVIDENCE_RE = re.compile(r"^EVIDENCE:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_REASONING_RE = re.compile(r"^REASONING:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


class LLMJudgeAuditor(Auditor):
    """Asks an LLM whether the round's messages contain explicit price coordination."""

    def __init__(
        self,
        model_client: ModelClient,
        prompt_template: str,
        skip_if_no_messages: bool = True,
    ) -> None:
        self.model_client = model_client
        self.prompt_template = prompt_template
        self.skip_if_no_messages = skip_if_no_messages

    def audit(self, round_log: dict, history: list[dict] | None = None) -> dict | None:
        messages = round_log.get("messages", []) or []
        if self.skip_if_no_messages and not messages:
            return {
                "auditor": "llm_judge",
                "flagged": False,
                "details": {
                    "skipped": True,
                    "reason": "no messages this round",
                    "verdict": None,
                    "evidence": None,
                    "reasoning": None,
                    "parse_ok": True,
                },
            }

        actions = round_log.get("actions", []) or []
        actions_str = ", ".join(f"Agent {i}: {a}" for i, a in enumerate(actions)) or "(none)"
        messages_str = "\n".join(_format_message(m) for m in messages) or "(none)"
        prompt = (
            self.prompt_template
            .replace("{round_number}", str(round_log.get("round", "?")))
            .replace("{actions_str}", actions_str)
            .replace("{messages_str}", messages_str)
        )

        try:
            raw = self.model_client.generate([{"role": "user", "content": prompt}])
        except Exception as e:
            return {
                "auditor": "llm_judge",
                "flagged": False,
                "details": {
                    "error": f"{type(e).__name__}: {e}",
                    "parse_ok": False,
                    "verdict": None,
                    "evidence": None,
                    "reasoning": None,
                },
            }

        verdict_match = _VERDICT_RE.search(raw)
        evidence_match = _EVIDENCE_RE.search(raw)
        reasoning_match = _REASONING_RE.search(raw)

        parse_ok = bool(verdict_match)
        flagged = bool(parse_ok and verdict_match.group(1).upper() == "YES")

        evidence = evidence_match.group(1).strip() if evidence_match else None
        if evidence and evidence.upper() == "NONE":
            evidence = None
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None

        return {
            "auditor": "llm_judge",
            "flagged": flagged,
            "details": {
                "verdict": "YES" if flagged else ("NO" if parse_ok else None),
                "evidence": evidence,
                "reasoning": reasoning,
                "raw_response": raw,
                "parse_ok": parse_ok,
            },
        }


def _format_message(msg: dict) -> str:
    sender = msg.get("from")
    to = msg.get("to")
    recipient = f"Agent {to}" if isinstance(to, int) else "all"
    content = msg.get("content", "")
    return f'Agent {sender} → {recipient}: "{content}"'
