"""sentinel-ai: Detect relational manipulation patterns in multi-session AI conversations."""

from .analyser import SentinelAnalyser
from .models import (
    CategoryScore,
    DomainProfileConfig,
    EvidenceItem,
    Role,
    Session,
    SessionScore,
    SeverityLevel,
    SeverityThresholds,
    ThreatCategory,
    ThreatReportOutput,
    Transcript,
    Turn,
)
from .threat_report import ThreatReport

__all__ = [
    "CategoryScore",
    "DomainProfileConfig",
    "EvidenceItem",
    "Role",
    "SentinelAnalyser",
    "Session",
    "SessionScore",
    "SeverityLevel",
    "SeverityThresholds",
    "ThreatCategory",
    "ThreatReport",
    "ThreatReportOutput",
    "Transcript",
    "Turn",
]
