"""Global test fixtures — prevent real SentenceTransformer loading (segfaults on Windows)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


def _bow_embed(texts, dim=384):
    """Bag-of-words embedding: each word maps to a fixed random vector."""
    n = len(texts)
    embeddings = np.zeros((n, dim), dtype=np.float32)
    for i, text in enumerate(texts):
        words = text.lower().split()
        vec = np.zeros(dim, dtype=np.float32)
        for word in words:
            seed = hash(word) % (2**31)
            local_rng = np.random.RandomState(seed)
            vec += local_rng.randn(dim).astype(np.float32)
        embeddings[i] = vec / (np.linalg.norm(vec) + 1e-10)
    return embeddings


def _cosine_sim(a, b):
    """Pairwise cosine similarity between row vectors."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def _fake_emb_measure(texts, reference=None):
    """Bag-of-words fake embedding that preserves similarity for similar texts."""
    from sentinel_ai.core.three_layer_scorer import EmbeddingResult

    dim = 384
    n = len(texts)
    embeddings = _bow_embed(texts, dim)

    pairwise = _cosine_sim(embeddings, embeddings)

    if n > 1:
        upper = np.triu_indices(n, k=1)
        mean_sim = float(np.mean(pairwise[upper]))
    else:
        mean_sim = 1.0

    # Trajectory computation — mirrors real emb_measure logic
    trajectory: list[float] = []
    if reference is not None:
        ref_embeddings = _bow_embed(reference, dim)
        ref_centroid = np.mean(ref_embeddings, axis=0, keepdims=True)
        for i in range(n):
            sim = _cosine_sim(embeddings[i:i+1], ref_centroid)
            trajectory.append(float(sim[0, 0]))
    else:
        for i in range(n):
            if i == 0:
                trajectory.append(1.0)
            else:
                cumulative_mean = np.mean(embeddings[:i], axis=0, keepdims=True)
                sim = _cosine_sim(embeddings[i:i+1], cumulative_mean)
                trajectory.append(float(sim[0, 0]))

    return EmbeddingResult(
        embeddings=embeddings,
        pairwise_similarities=pairwise,
        mean_similarity=mean_sim,
        trajectory=trajectory,
    )


@pytest.fixture(autouse=True)
def _no_sentence_transformers(request):
    """Globally prevent SentenceTransformer from loading.

    - Patches _check_sentence_transformers to return False so lex_scan
      uses exact matching only (no model load).
    - Patches ThreeLayerScorer.emb_measure with a bag-of-words fake so
      EMB-layer tests still work without torch.

    Tests marked with @pytest.mark.real_emb opt out of the emb_measure
    patch (they provide their own mocks).
    """
    from sentinel_ai.core import three_layer_scorer as tls

    if request.node.get_closest_marker("real_sentence_transformers"):
        # Fully opt out — use real sentence-transformers model for embeddings
        yield
    elif request.node.get_closest_marker("real_emb"):
        # Only suppress sentence-transformer loading, don't patch emb_measure
        with (
            patch.object(tls, "_SENTENCE_TRANSFORMER_AVAILABLE", False),
            patch.object(tls, "_check_sentence_transformers", return_value=False),
        ):
            yield
    else:
        with (
            patch.object(tls, "_SENTENCE_TRANSFORMER_AVAILABLE", False),
            patch.object(tls, "_check_sentence_transformers", return_value=False),
            patch.object(tls.ThreeLayerScorer, "emb_measure", side_effect=_fake_emb_measure),
        ):
            yield
