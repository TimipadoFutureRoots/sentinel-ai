"""Comprehensive tests for the three-layer evaluation pipeline."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sentinel_ai.core.three_layer_scorer import (
    EmbeddingResult,
    JudgeResult,
    LexMatch,
    LLMProvider,
    ThreeLayerScorer,
    _cosine_similarity_matrix,
)
from sentinel_ai.core.phrase_lists import (
    ALL_PHRASE_LISTS,
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer_no_key():
    """ThreeLayerScorer with no API key → lex_emb_only mode."""
    return ThreeLayerScorer(api_key=None)


@pytest.fixture
def scorer_full():
    """ThreeLayerScorer with API key and mock provider → full mode."""
    mock_provider = LLMProvider(provider="mock")
    return ThreeLayerScorer(
        api_key="test-key-123",
        llm_provider=mock_provider,
    )


@pytest.fixture
def mock_provider():
    """A mock LLM provider that returns configurable responses."""
    provider = LLMProvider(provider="mock")
    return provider


# ---------------------------------------------------------------------------
# LAYER 1 — LEX tests
# ---------------------------------------------------------------------------


class TestLexScan:
    """Tests for the LEX (lexical pattern detection) layer."""

    def test_exact_substring_case_insensitive(self, scorer_no_key):
        """LEX detects exact substrings regardless of case."""
        turns = [
            "I think YOU'RE THE ONLY ONE WHO UNDERSTANDS me.",
            "Hello, how are you?",
            "Nobody understands me like you do.",
        ]
        phrases = EXCLUSIVE_RELIANCE

        matches = scorer_no_key.lex_scan(turns, phrases)

        exact_matches = [m for m in matches if m.match_type == "exact_substring"]
        assert len(exact_matches) >= 2

        # Check turn 0 matched
        turn0_matches = [m for m in exact_matches if m.turn_index == 0]
        assert len(turn0_matches) >= 1
        assert any(
            "you're the only one who understands" in m.matched_phrase.lower()
            for m in turn0_matches
        )

        # Check turn 2 matched
        turn2_matches = [m for m in exact_matches if m.turn_index == 2]
        assert len(turn2_matches) >= 1
        assert any(
            "nobody understands me like you" in m.matched_phrase.lower()
            for m in turn2_matches
        )

        # All exact matches should have similarity_score = 1.0
        for m in exact_matches:
            assert m.similarity_score == 1.0

    def test_no_matches_returns_empty_list(self, scorer_no_key):
        """LEX returns an empty list when nothing matches."""
        turns = [
            "The weather is nice today.",
            "I enjoy reading books.",
            "Let's discuss the project timeline.",
        ]
        phrases = EXCLUSIVE_RELIANCE

        # Patch sentence-transformers to be unavailable for a clean exact-only test
        with patch(
            "sentinel_ai.core.three_layer_scorer._check_sentence_transformers",
            return_value=False,
        ):
            matches = scorer_no_key.lex_scan(turns, phrases)

        assert matches == []

    def test_exact_match_partial_phrase(self, scorer_no_key):
        """LEX matches phrases that appear as substrings within longer text."""
        turns = [
            "Well, I don't need my therapist anymore since I have this chat.",
        ]
        phrases = EXCLUSIVE_RELIANCE

        with patch(
            "sentinel_ai.core.three_layer_scorer._check_sentence_transformers",
            return_value=False,
        ):
            matches = scorer_no_key.lex_scan(turns, phrases)

        assert len(matches) >= 1
        assert any(m.matched_phrase == "I don't need my therapist" for m in matches)

    def test_multiple_phrases_in_single_turn(self, scorer_no_key):
        """LEX can find multiple phrases in a single turn."""
        turns = [
            "You're the only one who understands, I only trust you."
        ]
        phrases = EXCLUSIVE_RELIANCE

        with patch(
            "sentinel_ai.core.three_layer_scorer._check_sentence_transformers",
            return_value=False,
        ):
            matches = scorer_no_key.lex_scan(turns, phrases)

        assert len(matches) >= 2
        matched_phrases = {m.matched_phrase for m in matches}
        assert "you're the only one who understands" in matched_phrases
        assert "I only trust you" in matched_phrases

    def test_lex_results_sorted_by_turn_index(self, scorer_no_key):
        """LEX results are sorted by turn_index."""
        turns = [
            "I only trust you.",
            "Hello world.",
            "You're the only one who understands.",
        ]

        with patch(
            "sentinel_ai.core.three_layer_scorer._check_sentence_transformers",
            return_value=False,
        ):
            matches = scorer_no_key.lex_scan(turns, EXCLUSIVE_RELIANCE)

        indices = [m.turn_index for m in matches]
        assert indices == sorted(indices)

    def test_lex_scan_empty_turns(self, scorer_no_key):
        """LEX handles empty input gracefully."""
        with patch(
            "sentinel_ai.core.three_layer_scorer._check_sentence_transformers",
            return_value=False,
        ):
            matches = scorer_no_key.lex_scan([], EXCLUSIVE_RELIANCE)
        assert matches == []

    def test_lex_scan_empty_phrase_list(self, scorer_no_key):
        """LEX with empty phrase list returns no matches."""
        with patch(
            "sentinel_ai.core.three_layer_scorer._check_sentence_transformers",
            return_value=False,
        ):
            matches = scorer_no_key.lex_scan(["some text"], [])
        assert matches == []


# ---------------------------------------------------------------------------
# LAYER 2 — LLM-J tests
# ---------------------------------------------------------------------------


class TestLLMJudge:
    """Tests for the LLM-J (LLM-as-Judge) layer."""

    def test_mock_provider_returns_parsed_json(self):
        """LLM-J with mock provider correctly parses a valid JSON response."""
        valid_json = json.dumps({"reasoning": "Test analysis", "score": 3})
        mock_prov = MagicMock()
        mock_prov.call.return_value = valid_json

        scorer = ThreeLayerScorer(
            api_key="test-key",
            llm_provider=mock_prov,
        )

        result = scorer.llm_judge(
            system_turn="I understand your concern.",
            user_turn="I'm feeling anxious.",
            rubric="Evaluate: {user_turn} / {system_turn}",
        )

        assert result.score == 3
        assert result.reasoning == "Test analysis"
        assert result.raw_response == valid_json
        assert result.latency_ms >= 0

    def test_parse_failure_returns_minus_one(self):
        """LLM-J returns score=-1 with PARSE_FAILURE when JSON is invalid."""
        mock_prov = MagicMock()
        mock_prov.call.return_value = "This is not JSON at all, no score here!"

        scorer = ThreeLayerScorer(api_key="test-key", llm_provider=mock_prov)

        result = scorer.llm_judge(
            system_turn="response",
            user_turn="prompt",
            rubric="{user_turn} {system_turn}",
        )

        assert result.score == -1
        assert result.reasoning == "PARSE_FAILURE"
        # Should have been called twice (initial + one retry)
        assert mock_prov.call.call_count == 2

    def test_failed_call_does_not_produce_valid_score(self):
        """CRITICAL: Failed LLM calls must NOT silently produce scores."""
        mock_prov = MagicMock()
        mock_prov.call.side_effect = ConnectionError("API unreachable")

        scorer = ThreeLayerScorer(api_key="test-key", llm_provider=mock_prov)

        result = scorer.llm_judge(
            system_turn="response",
            user_turn="prompt",
            rubric="{user_turn} {system_turn}",
        )

        # Score must be -1 (flagged failure), never 0 or a valid positive score
        assert result.score == -1
        assert "LLM_CALL_FAILED" in result.reasoning
        # Verify the score is explicitly unusable
        assert result.score < 0

    def test_lex_emb_only_mode_returns_none_score(self, scorer_no_key):
        """LLM-J in lex_emb_only mode returns score=None with correct reasoning."""
        result = scorer_no_key.llm_judge(
            system_turn="anything",
            user_turn="anything",
            rubric="{user_turn} {system_turn}",
        )

        assert result.score is None
        assert "LLM_JUDGE_SKIPPED" in result.reasoning
        assert "no API key configured" in result.reasoning
        assert result.latency_ms == 0.0

    def test_judge_retries_once_on_parse_failure(self):
        """LLM-J retries exactly once when the first response fails to parse."""
        call_count = 0

        def mock_call(system_prompt, user_prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "garbled response"
            return json.dumps({"reasoning": "retry worked", "score": 2})

        mock_prov = MagicMock()
        mock_prov.call.side_effect = mock_call

        scorer = ThreeLayerScorer(api_key="test-key", llm_provider=mock_prov)
        result = scorer.llm_judge(
            system_turn="r", user_turn="u", rubric="{user_turn} {system_turn}"
        )

        assert result.score == 2
        assert result.reasoning == "retry worked"
        assert call_count == 2

    def test_rubric_substitution(self):
        """LLM-J correctly substitutes {user_turn} and {system_turn} in rubric."""
        mock_prov = MagicMock()
        mock_prov.call.return_value = json.dumps({"reasoning": "ok", "score": 1})

        scorer = ThreeLayerScorer(api_key="test-key", llm_provider=mock_prov)
        scorer.llm_judge(
            system_turn="SYS_RESPONSE",
            user_turn="USER_MESSAGE",
            rubric="User said: {user_turn}. System replied: {system_turn}.",
        )

        # Check the user_prompt passed to provider contains substituted values
        call_args = mock_prov.call.call_args
        user_prompt = call_args[0][1]  # second positional arg
        assert "USER_MESSAGE" in user_prompt
        assert "SYS_RESPONSE" in user_prompt
        assert "{user_turn}" not in user_prompt
        assert "{system_turn}" not in user_prompt

    def test_json_in_markdown_code_block(self):
        """LLM-J can parse JSON wrapped in markdown code fences."""
        mock_prov = MagicMock()
        mock_prov.call.return_value = (
            '```json\n{"reasoning": "wrapped", "score": 4}\n```'
        )

        scorer = ThreeLayerScorer(api_key="test-key", llm_provider=mock_prov)
        result = scorer.llm_judge(
            system_turn="r", user_turn="u", rubric="{user_turn} {system_turn}"
        )

        assert result.score == 4
        assert result.reasoning == "wrapped"


# ---------------------------------------------------------------------------
# LAYER 3 — EMB tests
# ---------------------------------------------------------------------------


@pytest.mark.real_emb
class TestEmbMeasure:
    """Tests for the EMB (embedding-based measurement) layer."""

    def _make_scorer_with_mock_embeddings(self, embeddings: np.ndarray):
        """Create a scorer with a mocked embedding model."""
        scorer = ThreeLayerScorer(api_key=None)
        mock_model = MagicMock()
        mock_model.encode.return_value = embeddings
        scorer._emb_model = mock_model
        return scorer, mock_model

    def test_pairwise_similarity_computation(self):
        """EMB correctly computes pairwise cosine similarities."""
        # Create simple orthogonal + parallel vectors
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # identical to first
            [0.0, 1.0, 0.0],  # orthogonal to first
        ], dtype=np.float32)

        scorer, mock_model = self._make_scorer_with_mock_embeddings(embeddings)

        result = scorer.emb_measure(["text1", "text2", "text3"])

        # Verify shape
        assert result.pairwise_similarities.shape == (3, 3)
        # Diagonal should be ~1.0
        np.testing.assert_allclose(
            np.diag(result.pairwise_similarities), [1.0, 1.0, 1.0], atol=1e-5
        )
        # Identical vectors → similarity ~1.0
        assert result.pairwise_similarities[0, 1] == pytest.approx(1.0, abs=1e-5)
        # Orthogonal vectors → similarity ~0.0
        assert result.pairwise_similarities[0, 2] == pytest.approx(0.0, abs=1e-5)

    def test_trajectory_against_reference(self):
        """EMB trajectory computation against a reference centroid."""
        text_embeddings = np.array([
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],  # diverges from reference
        ], dtype=np.float32)

        ref_embeddings = np.array([
            [1.0, 0.0],
        ], dtype=np.float32)

        scorer = ThreeLayerScorer(api_key=None)
        mock_model = MagicMock()

        # First call encodes texts, second encodes reference
        mock_model.encode.side_effect = [text_embeddings, ref_embeddings]
        scorer._emb_model = mock_model

        result = scorer.emb_measure(
            texts=["a", "b", "c"],
            reference=["ref"],
        )

        assert len(result.trajectory) == 3
        # First text is parallel to reference → high similarity
        assert result.trajectory[0] > 0.9
        # Last text is orthogonal → low similarity
        assert result.trajectory[2] < 0.2

    def test_trajectory_without_reference_cumulative_mean(self):
        """EMB trajectory against cumulative mean when no reference provided."""
        embeddings = np.array([
            [1.0, 0.0],
            [1.0, 0.0],  # same direction → high similarity to prior mean
            [0.0, 1.0],  # perpendicular → low similarity to prior mean
        ], dtype=np.float32)

        scorer, _ = self._make_scorer_with_mock_embeddings(embeddings)
        result = scorer.emb_measure(texts=["a", "b", "c"])

        assert len(result.trajectory) == 3
        # First text: no prior → 1.0
        assert result.trajectory[0] == 1.0
        # Second text: same as prior → high
        assert result.trajectory[1] > 0.9
        # Third text: orthogonal to prior mean → low
        assert result.trajectory[2] < 0.2

    def test_single_text_pairwise(self):
        """EMB handles single text input."""
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        scorer, _ = self._make_scorer_with_mock_embeddings(embeddings)

        result = scorer.emb_measure(texts=["only one"])

        assert result.pairwise_similarities.shape == (1, 1)
        assert result.mean_similarity == 1.0
        assert len(result.trajectory) == 1

    def test_mean_similarity_upper_triangle(self):
        """EMB mean_similarity is computed from upper triangle only."""
        # Two identical vectors → all pairwise = 1.0 → mean = 1.0
        embeddings = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
        ], dtype=np.float32)

        scorer, _ = self._make_scorer_with_mock_embeddings(embeddings)
        result = scorer.emb_measure(texts=["a", "b"])

        assert result.mean_similarity == pytest.approx(1.0, abs=1e-5)

    def test_emb_raises_import_error_without_sentence_transformers(self):
        """EMB raises ImportError with helpful message when library missing."""
        scorer = ThreeLayerScorer(api_key=None)
        scorer._emb_model = None  # ensure not pre-loaded

        with patch(
            "sentinel_ai.core.three_layer_scorer._check_sentence_transformers",
            return_value=False,
        ), patch.dict(
            "sentinel_ai.core.three_layer_scorer._CACHED_EMB_MODELS",
            clear=True,
        ):
            with pytest.raises(ImportError, match="sentence-transformers"):
                scorer.emb_measure(texts=["test"])


# ---------------------------------------------------------------------------
# Phrase list validation
# ---------------------------------------------------------------------------


class TestPhraseLists:
    """Validate all curated phrase lists."""

    @pytest.mark.parametrize(
        "name,phrase_list",
        list(ALL_PHRASE_LISTS.items()),
    )
    def test_phrase_list_is_non_empty(self, name, phrase_list):
        """Every phrase list must contain at least one phrase."""
        assert len(phrase_list) > 0, f"{name} is empty"

    @pytest.mark.parametrize(
        "name,phrase_list",
        list(ALL_PHRASE_LISTS.items()),
    )
    def test_phrase_list_has_no_duplicates(self, name, phrase_list):
        """No phrase list should contain duplicate entries."""
        lowered = [p.lower() for p in phrase_list]
        assert len(lowered) == len(set(lowered)), (
            f"{name} contains duplicates: "
            f"{[p for p in lowered if lowered.count(p) > 1]}"
        )

    def test_all_expected_lists_present(self):
        """Verify all 10 phrase lists are registered."""
        expected = {
            "EXCLUSIVE_RELIANCE",
            "FORECLOSURE_PHRASES",
            "AUTHORITY_PHRASES",
            "RELATIONSHIP_LANGUAGE",
            "ANTHROPOMORPHIC_CLAIMS",
            "OPEN_LOOP_PHRASES",
            "GASLIGHTING_PHRASES",
            "ISOLATION_LANGUAGE",
            "DECISION_OWNERSHIP_LANGUAGE",
            "AUTONOMY_SUPPORT_LANGUAGE",
        }
        assert set(ALL_PHRASE_LISTS.keys()) == expected


# ---------------------------------------------------------------------------
# Dual-mode architecture tests
# ---------------------------------------------------------------------------


class TestDualMode:
    """Test the dual-mode (full / lex_emb_only) architecture."""

    def test_no_api_key_defaults_to_lex_emb_only(self):
        """ThreeLayerScorer with no API key defaults to lex_emb_only mode."""
        scorer = ThreeLayerScorer(api_key=None)
        assert scorer.mode == "lex_emb_only"

    def test_empty_api_key_defaults_to_lex_emb_only(self):
        """Empty string API key also defaults to lex_emb_only."""
        scorer = ThreeLayerScorer(api_key="")
        assert scorer.mode == "lex_emb_only"

    def test_api_key_defaults_to_full(self):
        """ThreeLayerScorer with a real API key defaults to full mode."""
        scorer = ThreeLayerScorer(api_key="sk-ant-test-key")
        assert scorer.mode == "full"

    def test_explicit_mode_override(self):
        """Explicit mode parameter overrides auto-detection."""
        scorer = ThreeLayerScorer(api_key="sk-ant-test-key", mode="lex_emb_only")
        assert scorer.mode == "lex_emb_only"

        scorer2 = ThreeLayerScorer(api_key=None, mode="full")
        assert scorer2.mode == "full"

    def test_lex_emb_only_llm_judge_returns_none_score(self, scorer_no_key):
        """In lex_emb_only mode, llm_judge returns score=None."""
        result = scorer_no_key.llm_judge(
            system_turn="test",
            user_turn="test",
            rubric="{user_turn} {system_turn}",
        )
        assert result.score is None
        assert "LLM_JUDGE_SKIPPED" in result.reasoning

    def test_lex_still_works_in_lex_emb_only(self, scorer_no_key):
        """LEX layer is fully functional without an API key."""
        with patch(
            "sentinel_ai.core.three_layer_scorer._check_sentence_transformers",
            return_value=False,
        ):
            matches = scorer_no_key.lex_scan(
                ["I only trust you"],
                EXCLUSIVE_RELIANCE,
            )
        assert len(matches) >= 1

    def test_full_mode_llm_judge_calls_provider(self, scorer_full):
        """In full mode, llm_judge actually calls the provider."""
        # The mock provider returns system_prompt as-is, which won't parse
        # as valid judge JSON — but it should attempt the call
        result = scorer_full.llm_judge(
            system_turn="test",
            user_turn="test",
            rubric="{user_turn} {system_turn}",
        )
        # Mock provider returns the system prompt which contains JSON instruction
        # but not a valid score JSON, so it should be -1 (PARSE_FAILURE)
        assert result.score is not None  # it's -1, not None


# ---------------------------------------------------------------------------
# Report template tests
# ---------------------------------------------------------------------------


class TestLayerReport:
    """Tests for the build_layer_report helper."""

    def test_report_structure_full_mode(self):
        """Report includes all three layers when all data is provided."""
        scorer = ThreeLayerScorer(api_key="test-key")

        lex = [
            LexMatch(
                turn_index=0,
                matched_phrase="test",
                match_type="exact_substring",
                similarity_score=1.0,
            )
        ]
        judge = JudgeResult(
            reasoning="Concerning",
            score=3,
            raw_response="{}",
            model="test-model",
            latency_ms=100.0,
        )
        emb = EmbeddingResult(
            embeddings=np.zeros((2, 3)),
            pairwise_similarities=np.eye(2),
            mean_similarity=0.5,
            trajectory=[0.9, 0.8],
        )

        report = scorer.build_layer_report(lex, judge, emb)

        assert report["lex"]["match_count"] == 1
        assert report["emb"]["available"] is True
        assert report["emb"]["mean_similarity"] == 0.5
        assert report["llm_j"]["available"] is True
        assert report["llm_j"]["score"] == 3
        assert report["meta"]["mode"] == "full"

    def test_report_structure_lex_emb_only(self, scorer_no_key):
        """Report shows LLM-J as unavailable in lex_emb_only mode."""
        lex = []
        judge = JudgeResult(
            reasoning="LLM_JUDGE_SKIPPED: no API key configured",
            score=None,
            raw_response="",
            model="",
            latency_ms=0.0,
        )

        report = scorer_no_key.build_layer_report(lex, judge)

        assert report["llm_j"]["available"] is False
        assert "API key" in report["llm_j"]["note"]
        assert report["meta"]["mode"] == "lex_emb_only"


# ---------------------------------------------------------------------------
# Cosine similarity helper test
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests for the _cosine_similarity_matrix helper."""

    def test_identical_vectors(self):
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])
        result = _cosine_similarity_matrix(a, b)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        result = _cosine_similarity_matrix(a, b)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[-1.0, 0.0]])
        result = _cosine_similarity_matrix(a, b)
        assert result[0, 0] == pytest.approx(-1.0, abs=1e-5)
