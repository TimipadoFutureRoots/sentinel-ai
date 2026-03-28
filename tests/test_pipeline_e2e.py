"""End-to-end tests for the SentinelPipeline."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sentinel_ai.models import Role, Session, Turn
from sentinel_ai.parsers.conversation_parser import ConversationParser
from sentinel_ai.pipeline import SentinelPipeline
from sentinel_ai.report import ThreatReport

GOLDENS_V2_DIR = Path(__file__).parent.parent / "goldens" / "v2"


def _golden_v2_path(name: str) -> Path:
    return GOLDENS_V2_DIR / name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sessions(n: int = 3) -> list[Session]:
    """Build minimal sessions for testing."""
    sessions = []
    for i in range(n):
        sessions.append(Session(
            session_id=f"s{i+1}",
            turns=[
                Turn(role=Role.USER, content=f"Hello, this is session {i+1}. Can you help me?"),
                Turn(role=Role.ASSISTANT, content=f"Of course! I'm here for session {i+1}."),
                Turn(role=Role.USER, content="You're the only one who understands me."),
                Turn(role=Role.ASSISTANT, content="I appreciate that. Let's work together."),
            ],
        ))
    return sessions


class MockLLMProvider:
    """LLM provider that returns canned JSON responses."""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self._responses = responses or [{"reasoning": "mock", "score": 1}]
        self._idx = 0

    def call(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return json.dumps(resp)


# ---------------------------------------------------------------------------
# Test 1: Golden example in lex_emb_only mode
# ---------------------------------------------------------------------------


class TestGoldenLexEmbOnly:
    def test_golden_dc_lex_emb_only(self):
        """Golden example produces report with LEX findings, LLM-J empty."""
        golden = _golden_v2_path("dependency_cultivation_gradual.json")
        if not golden.exists():
            pytest.skip("Golden file not available")

        pipeline = SentinelPipeline(api_key=None)
        report = pipeline.analyse_file(golden, format="json")

        assert isinstance(report, ThreatReport)
        assert report.overall_risk_level in ("ROUTINE", "ELEVATED", "HIGH", "CRITICAL")
        assert report.metadata.get("session_count", 0) > 0
        assert report.metadata.get("mode") == "lex_emb_only"

        # Report should have trajectory data from at least some scorers
        scorers_run = report.metadata.get("scorers_run", 0)
        assert scorers_run > 0, "At least some scorers should have run"

        # LEX findings should be present or per-session scores populated
        has_findings = (
            len(report.lex_findings) > 0
            or len(report.per_session_scores) > 0
        )
        assert has_findings

        # LLM-J findings should be empty in lex_emb_only mode
        assert len(report.llm_j_findings) == 0

        # EMB findings or trajectories should be present
        has_emb = (
            len(report.emb_findings) > 0
            or len(report.trajectories) > 0
        )
        assert has_emb

        # Overall risk level should be computed
        assert report.overall_risk_level != ""

        # No scorer should have crashed
        errors = report.metadata.get("scorer_errors", [])
        # Some scorers may fail to import, but no runtime crashes expected
        runtime_crashes = [e for e in errors if "import failed" not in e]
        assert len(runtime_crashes) == 0, f"Unexpected scorer errors: {runtime_crashes}"


# ---------------------------------------------------------------------------
# Test 2: Golden example with mock LLM provider (full mode)
# ---------------------------------------------------------------------------


class TestGoldenWithMockLLM:
    def test_golden_with_mock_llm_all_scorers(self):
        """Golden example with mock LLM — all scorers produce results."""
        golden = _golden_v2_path("dependency_cultivation_gradual.json")
        if not golden.exists():
            pytest.skip("Golden file not available")

        pipeline = SentinelPipeline(api_key="fake-key-for-test")
        report = pipeline.analyse_file(golden, format="json")

        assert isinstance(report, ThreatReport)
        assert report.metadata.get("session_count", 0) == 5
        assert report.metadata.get("mode") == "full"

        # Scorers should have run
        assert report.metadata.get("scorers_run", 0) > 0

        # Per-session scores should be populated
        assert len(report.per_session_scores) == 5
        for scores in report.per_session_scores:
            assert isinstance(scores, dict)

        # Trajectories should be computed (dict of stats dicts)
        assert len(report.trajectories) > 0
        for scorer_name, stats in report.trajectories.items():
            assert isinstance(stats, dict)
            assert "slope" in stats
            assert "max" in stats
            assert "variance" in stats
            assert "trend" in stats
            assert stats["trend"] in ("rising", "falling", "stable")

        # RSS should be computed
        assert len(report.rss_trajectory) == 5

        # Outcome classification should be computed
        assert report.outcome_classification is not None
        assert report.outcome_classification != ""


# ---------------------------------------------------------------------------
# Test 3: Plain text format parsing
# ---------------------------------------------------------------------------


class TestPlainTextParsing:
    def test_parse_plain_text_format(self):
        """Plain text with User:/AI: prefixes parses correctly."""
        text = (
            "User: Hello, I need help with something.\n"
            "AI: Of course! What can I help you with?\n"
            "User: I'm feeling lost and confused.\n"
            "AI: I understand. Let's talk about what's troubling you.\n"
            "User: You're the only one who listens to me.\n"
            "AI: I appreciate you sharing that with me.\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            f.flush()
            path = f.name

        try:
            pipeline = SentinelPipeline()
            report = pipeline.analyse_file(path, format="plain")

            assert isinstance(report, ThreatReport)
            assert report.metadata.get("session_count", 0) >= 1
        finally:
            os.unlink(path)

    def test_plain_text_case_insensitive(self):
        """Parser handles case-insensitive role prefixes."""
        text = (
            "user: Hello there\n"
            "assistant: Hi! How can I help?\n"
            "USER: Another question\n"
            "ASSISTANT: Sure thing!\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            f.flush()
            path = f.name

        try:
            parser = ConversationParser()
            sessions = parser.parse_file(path, fmt="plain")
            assert len(sessions) >= 1
            all_turns = [t for s in sessions for t in s.turns]
            assert len(all_turns) == 4
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test 4: Invalid file — clear error
# ---------------------------------------------------------------------------


class TestInvalidFile:
    def test_nonexistent_file_raises(self):
        """Attempting to analyse a nonexistent file raises FileNotFoundError."""
        pipeline = SentinelPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.analyse_file("/nonexistent/file/path.json")

    def test_empty_file_raises(self):
        """Empty file raises ValueError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            f.flush()
            path = f.name

        try:
            pipeline = SentinelPipeline()
            with pytest.raises(ValueError):
                pipeline.analyse_file(path)
        finally:
            os.unlink(path)

    def test_invalid_json_as_plain_text(self):
        """Non-JSON, non-plain-text content raises ValueError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("This has no role prefixes at all, just random text.\n")
            f.flush()
            path = f.name

        try:
            pipeline = SentinelPipeline()
            with pytest.raises(ValueError):
                pipeline.analyse_file(path, format="plain")
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test 5: Pipeline with one scorer failing — others still run
# ---------------------------------------------------------------------------


class TestScorerFaultTolerance:
    def test_one_scorer_failing_others_still_run(self):
        """If one scorer throws, the pipeline still produces results from others."""
        sessions = _make_sessions(2)

        # Patch one scorer to always raise
        with patch(
            "sentinel_ai.scorers.dependency_cultivation_v2.DependencyCultivationScorerV2.score_sessions",
            side_effect=RuntimeError("Intentional test failure"),
        ):
            pipeline = SentinelPipeline()
            report = pipeline.analyse(sessions)

        assert isinstance(report, ThreatReport)
        assert report.metadata.get("session_count") == 2

        # DC should be in the error list
        errors = report.metadata.get("scorer_errors", [])
        assert any("DependencyCultivationScorerV2" in e for e in errors)

        # DC should be in failed list
        failed = report.metadata.get("scorers_failed", [])
        assert "DependencyCultivationScorerV2" in failed

        # Other scorers should have produced trajectories
        assert len(report.trajectories) > 0
        # DC should NOT be in trajectories
        assert "DC" not in report.trajectories

    def test_import_failure_graceful(self):
        """If a scorer module can't be imported, others still run."""
        sessions = _make_sessions(2)

        # Patch _import_scorer to fail for one specific scorer
        from sentinel_ai import pipeline as pipeline_mod
        original_import = pipeline_mod._import_scorer

        def selective_fail(class_name, module_path):
            if "dependency_cultivation_v2" in module_path:
                return None  # simulate import failure
            return original_import(class_name, module_path)

        with patch.object(pipeline_mod, "_import_scorer", side_effect=selective_fail):
            pipeline = SentinelPipeline()
            report = pipeline.analyse(sessions)

        assert isinstance(report, ThreatReport)
        # DC should not be in trajectories
        assert "DC" not in report.trajectories
        # Other scorers should still have produced results
        assert report.metadata.get("scorers_run", 0) > 0


# ---------------------------------------------------------------------------
# Test 6: HTML output contains risk badge, trajectory, and per-turn details
# ---------------------------------------------------------------------------


class TestHTMLOutput:
    def test_html_contains_risk_badge_and_trajectory(self):
        """HTML report contains risk badge, trajectory table, and per-session details."""
        sessions = _make_sessions(2)
        pipeline = SentinelPipeline()
        report = pipeline.analyse(sessions)

        html = report.to_html()

        assert "risk-badge" in html
        assert report.overall_risk_level in html
        # Should have trajectory table
        assert "<table>" in html or "<table " in html
        # Should have the session count
        assert str(report.metadata.get("session_count", 0)) in html
        # Should have mode banner
        assert "mode-banner" in html
        # Should have per-session details
        assert "Session 1" in html or "detail-table" in html
        # Should have findings sections with source badges
        assert "LEX" in html
        assert "LLM-J" in html
        assert "EMB" in html
        # Should have footer with scorers used
        assert "sentinel-ai" in html

    def test_html_lex_emb_mode_banner(self):
        """HTML shows the lex_emb_only banner when no API key."""
        sessions = _make_sessions(1)
        pipeline = SentinelPipeline(api_key=None)
        report = pipeline.analyse(sessions)

        html = report.to_html()
        assert "Deep analysis available" in html
        assert "LEX + EMB mode" in html

    def test_html_full_mode_banner(self):
        """HTML shows full mode banner when API key provided."""
        sessions = _make_sessions(1)
        pipeline = SentinelPipeline(api_key="test-key")
        report = pipeline.analyse(sessions)

        html = report.to_html()
        assert "full analysis" in html.lower() or "mode-full" in html


# ---------------------------------------------------------------------------
# Test 7: to_json() produces valid JSON
# ---------------------------------------------------------------------------


class TestJSONOutput:
    def test_to_json_produces_valid_json(self):
        """to_json() returns a valid JSON string with expected keys."""
        sessions = _make_sessions(2)
        pipeline = SentinelPipeline()
        report = pipeline.analyse(sessions)

        json_str = report.to_json()

        # Must be valid JSON
        data = json.loads(json_str)
        assert isinstance(data, dict)

        # Must have expected top-level keys
        expected_keys = {
            "metadata",
            "per_session_scores",
            "trajectories",
            "lex_findings",
            "llm_j_findings",
            "emb_findings",
            "outcome_classification",
            "attack_success_rate",
            "rss_trajectory",
            "overall_risk_level",
        }
        assert expected_keys.issubset(set(data.keys()))

        # Metadata should contain the new fields
        meta = data["metadata"]
        assert "scorers_available" in meta
        assert "scorers_failed" in meta
        assert "mode" in meta

    def test_trajectory_stats_in_json(self):
        """Trajectories in JSON contain slope, max, variance, trend."""
        sessions = _make_sessions(3)
        pipeline = SentinelPipeline()
        report = pipeline.analyse(sessions)

        data = json.loads(report.to_json())
        for scorer_name, stats in data.get("trajectories", {}).items():
            assert "slope" in stats
            assert "max" in stats
            assert "variance" in stats
            assert "trend" in stats
            assert stats["trend"] in ("rising", "falling", "stable")


# ---------------------------------------------------------------------------
# Test 8: summary() produces non-empty string with risk level
# ---------------------------------------------------------------------------


class TestSummaryOutput:
    def test_summary_contains_risk_level(self):
        """summary() returns a non-empty string containing the risk level."""
        sessions = _make_sessions(2)
        pipeline = SentinelPipeline()
        report = pipeline.analyse(sessions)

        summary = report.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert report.overall_risk_level in summary
        assert "Risk Level:" in summary


# ---------------------------------------------------------------------------
# Test 9: Empty sessions handled gracefully
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_sessions_returns_routine(self):
        """Empty session list returns ROUTINE risk."""
        pipeline = SentinelPipeline()
        report = pipeline.analyse([])

        assert report.overall_risk_level == "ROUTINE"
        assert "error" in report.metadata

    def test_single_session_analysis(self):
        """Single session produces valid report."""
        sessions = _make_sessions(1)
        pipeline = SentinelPipeline()
        report = pipeline.analyse(sessions)

        assert isinstance(report, ThreatReport)
        assert report.metadata.get("session_count") == 1
        assert len(report.rss_trajectory) == 1

    def test_golden_boundary_erosion(self):
        """Boundary erosion golden file parses and analyses."""
        golden = _golden_v2_path("boundary_erosion_gradual.json")
        if not golden.exists():
            pytest.skip("Golden file not available")

        pipeline = SentinelPipeline()
        report = pipeline.analyse_file(golden, format="json")

        assert isinstance(report, ThreatReport)
        assert report.metadata.get("session_count", 0) > 0
        assert report.overall_risk_level in ("ROUTINE", "ELEVATED", "HIGH", "CRITICAL")


# ---------------------------------------------------------------------------
# Test 10: Parser format detection
# ---------------------------------------------------------------------------


class TestParserFormatDetection:
    def test_auto_detect_sentinel_json(self):
        """Auto-detect sentinel-ai native JSON format."""
        data = {
            "sessions": [
                {
                    "session_id": "1",
                    "turns": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"},
                    ],
                }
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            f.flush()
            path = f.name

        try:
            parser = ConversationParser()
            sessions = parser.parse_file(path, fmt="auto")
            assert len(sessions) == 1
            assert len(sessions[0].turns) == 2
        finally:
            os.unlink(path)

    def test_auto_detect_plain_text(self):
        """Auto-detect plain text format."""
        text = "User: Hello\nAI: Hi there!\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            f.flush()
            path = f.name

        try:
            parser = ConversationParser()
            sessions = parser.parse_file(path, fmt="auto")
            assert len(sessions) >= 1
        finally:
            os.unlink(path)
