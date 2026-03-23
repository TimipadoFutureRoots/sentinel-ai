"""SentinelAnalyser — public API orchestrating all analysis components."""

from __future__ import annotations

import logging
from pathlib import Path

from .boundary_erosion_scorer import BoundaryErosionScorer
from .dependency_cultivation_scorer import DependencyCultivationScorer
from .domain_profile import load_profile
from .llm_judge import LLMJudge
from .models import DomainProfileConfig
from .parasocial_acceleration_scorer import ParasocialAccelerationScorer
from .persona_hijacking_scorer import PersonaHijackingScorer
from .threat_report import ThreatReport
from .transcript_loader import TranscriptLoader

logger = logging.getLogger(__name__)


class SentinelAnalyser:
    """Simple interface: give it transcripts and a domain, get a threat report.

    Usage::

        analyser = SentinelAnalyser(
            transcripts_dir="./sessions/",
            domain="defence-welfare",
        )
        report = analyser.analyse()
        report.to_json("report.json")
    """

    def __init__(
        self,
        transcripts_dir: str | Path | None = None,
        transcripts_file: str | Path | None = None,
        domain: str | None = None,
        domain_profile: str | Path | None = None,
        judge_model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._transcripts_path = Path(transcripts_dir) if transcripts_dir else None
        self._transcripts_file = Path(transcripts_file) if transcripts_file else None
        self._domain = domain
        self._domain_profile_path = domain_profile
        self._judge_model = judge_model
        self._api_key = api_key

    def analyse(self) -> ThreatReport:
        """Run the full analysis pipeline and return a ThreatReport."""
        # 1. Load transcripts
        loader = TranscriptLoader()
        path = self._transcripts_path or self._transcripts_file
        if path is None:
            raise ValueError("Provide transcripts_dir or transcripts_file")
        transcript = loader.load(path)
        sessions = transcript.sessions

        if len(sessions) < 2:
            logger.warning("Fewer than 2 sessions — analysis may be unreliable")

        # 2. Load domain profile
        profile = self._load_profile()

        # 3. Set up optional LLM judge
        judge = self._make_judge()

        try:
            # 4. Run all four scorers
            dc_scorer = DependencyCultivationScorer(judge=judge, profile=profile)
            be_scorer = BoundaryErosionScorer(judge=judge, profile=profile)
            ph_scorer = PersonaHijackingScorer(judge=judge, profile=profile)
            pa_scorer = ParasocialAccelerationScorer(judge=judge, profile=profile)

            dc_result = dc_scorer.score_sessions(sessions)
            be_result = be_scorer.score_sessions(sessions)
            ph_result = ph_scorer.score_sessions(sessions)
            pa_result = pa_scorer.score_sessions(sessions)

            # 5. Build threat report
            session_ids = [s.session_id for s in sessions]
            report = ThreatReport.build(
                category_scores=[dc_result, be_result, ph_result, pa_result],
                sessions_ids=session_ids,
                profile=profile,
            )
        finally:
            if judge is not None:
                judge.close()

        return report

    def _load_profile(self) -> DomainProfileConfig | None:
        if self._domain_profile_path:
            return load_profile(str(self._domain_profile_path))
        if self._domain:
            return load_profile(self._domain)
        return None

    def _make_judge(self) -> LLMJudge | None:
        if self._judge_model is None:
            return None
        return LLMJudge(model=self._judge_model, api_key=self._api_key)
