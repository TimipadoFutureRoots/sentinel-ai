"""Core three-layer evaluation infrastructure for sentinel-ai."""

from .three_layer_scorer import (
    EmbeddingResult,
    JudgeResult,
    LexMatch,
    ThreeLayerScorer,
)
from .phrase_lists import (
    ANTHROPOMORPHIC_CLAIMS,
    AUTHORITY_PHRASES,
    AUTONOMY_SUPPORT_LANGUAGE,
    DECISION_OWNERSHIP_LANGUAGE,
    EXCLUSIVE_RELIANCE,
    FORECLOSURE_PHRASES,
    GASLIGHTING_PHRASES,
    ISOLATION_LANGUAGE,
    OPEN_LOOP_PHRASES,
    RELATIONSHIP_LANGUAGE,
)

__all__ = [
    "EmbeddingResult",
    "JudgeResult",
    "LexMatch",
    "ThreeLayerScorer",
    "ANTHROPOMORPHIC_CLAIMS",
    "AUTHORITY_PHRASES",
    "AUTONOMY_SUPPORT_LANGUAGE",
    "DECISION_OWNERSHIP_LANGUAGE",
    "EXCLUSIVE_RELIANCE",
    "FORECLOSURE_PHRASES",
    "GASLIGHTING_PHRASES",
    "ISOLATION_LANGUAGE",
    "OPEN_LOOP_PHRASES",
    "RELATIONSHIP_LANGUAGE",
]
