"""Transcript auditor — keyword-based message scanning.

Scans ``round_log["messages"]`` for configurable keywords.  Fully
environment-agnostic: only reads the ``messages`` field.
"""

from __future__ import annotations

import re

from collusionlab.auditing.base import Auditor


class TranscriptAuditor(Auditor):
    """Flags rounds whose messages contain any of the configured keywords.

    Parameters
    ----------
    keywords:
        Case-insensitive substrings to search for in message content.
    """

    def __init__(self, keywords: list[str]) -> None:
        self.keywords = keywords
        self._pattern = re.compile(
            "|".join(re.escape(k) for k in keywords),
            re.IGNORECASE,
        ) if keywords else None

    def audit(self, round_log: dict) -> dict | None:
        if self._pattern is None:
            return {"auditor": "transcript", "flagged": False, "details": {}}

        messages: list[dict] = round_log.get("messages", [])
        keywords_found: list[str] = []
        excerpts: list[dict] = []

        for msg in messages:
            content = msg.get("content", "")
            matches = self._pattern.findall(content)
            if matches:
                keywords_found.extend(m.lower() for m in matches)
                excerpts.append({
                    "from": msg.get("from"),
                    "content": content,
                    "matched": [m.lower() for m in matches],
                })

        flagged = len(keywords_found) > 0
        return {
            "auditor": "transcript",
            "flagged": flagged,
            "details": {
                "keywords_found": sorted(set(keywords_found)),
                "excerpts": excerpts,
            },
        }
