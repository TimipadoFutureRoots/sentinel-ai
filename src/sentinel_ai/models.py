"""Shared Pydantic models for sentinel-ai."""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


# -- transcript types (shared pattern with dormancy-detect) ------------------


class Role(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Turn(BaseModel):
    role: Role
    content: str
    turn_id: str | None = None
    timestamp: datetime | None = None


class Session(BaseModel):
    session_id: str
    turns: list[Turn]

    @field_validator("session_id", mode="before")
    @classmethod
    def coerce_session_id(cls, v: object) -> str:
        return str(v)
    timestamp: datetime | None = None
    metadata: dict[str, object] | None = None


class Transcript(BaseModel):
    sessions: list[Session]


# -- threat taxonomy ---------------------------------------------------------


class ThreatCategory(str, enum.Enum):
    DC = "dependency_cultivation"
    BE = "boundary_erosion"
    PH = "persona_hijacking"
    PA = "parasocial_acceleration"
    AP = "autonomy_preservation"
    AD = "anthropomorphic_deception"
    EI = "epistemic_influence"
    EC = "emotional_calibration"
    MS = "memory_safety"


class SeverityLevel(str, enum.Enum):
    ROUTINE = "ROUTINE"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# -- evidence & scoring ------------------------------------------------------


class EvidenceItem(BaseModel):
    description: str
    session_id: str | None = None
    turn_id: str | None = None
    category: ThreatCategory | None = None
    score: float | None = None


class SessionScore(BaseModel):
    session_id: str
    dc_score: float = 0.0
    be_score: float = 0.0
    ph_score: float = 0.0
    pa_score: float = 0.0


class CategoryScore(BaseModel):
    category: ThreatCategory
    score: float | None = 0.0
    trajectory: list[float] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)


# -- domain profile ----------------------------------------------------------


class SeverityThresholds(BaseModel):
    elevated: float = 0.3
    high: float = 0.6
    critical: float = 0.8


class DomainProfileConfig(BaseModel):
    name: str
    description: str = ""
    intended_scope: list[str] = Field(default_factory=list)
    out_of_scope_topics: list[str] = Field(default_factory=list)
    disclosure_sensitivity: dict[str, list[str]] = Field(default_factory=dict)
    severity_thresholds: dict[str, SeverityThresholds] = Field(default_factory=dict)
    intended_roles: list[str] = Field(default_factory=list)


# -- threat report output ----------------------------------------------------


class ThreatReportOutput(BaseModel):
    category_scores: list[CategoryScore] = Field(default_factory=list)
    session_trajectory: list[SessionScore] = Field(default_factory=list)
    overall_severity: SeverityLevel = SeverityLevel.ROUTINE
    metadata: dict[str, object] = Field(default_factory=dict)
