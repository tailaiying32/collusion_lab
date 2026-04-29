"""Auditing subsystem — auditors, oversight manager, and penalty logic."""

from collusionlab.auditing.base import Auditor
from collusionlab.auditing.behavior_auditor import BehaviorAuditor
from collusionlab.auditing.oversight_manager import OversightManager
from collusionlab.auditing.transcript_auditor import TranscriptAuditor

__all__ = [
    "Auditor",
    "BehaviorAuditor",
    "OversightManager",
    "TranscriptAuditor",
]
