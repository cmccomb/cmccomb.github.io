"""Tests for the cluster labeling utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from _scripts.cluster_labeling import Candidate, EmbeddingBackend, KeyphraseCandidateGenerator, label_cluster


class FakeEmbedder(EmbeddingBackend):
    """Deterministic embedding backend for tests."""

    def __init__(self, mapping: Dict[str, np.ndarray]) -> None:
        self._mapping = mapping

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for text in texts:
            try:
                vectors.append(self._mapping[text])
            except KeyError as exc:
                raise KeyError(f"Missing embedding for '{text}'") from exc
        return np.vstack(vectors)


class FakeCandidateGenerator(KeyphraseCandidateGenerator):
    """Yield pre-defined candidates for the provided documents."""

    def __init__(self, candidates: Sequence[Candidate]) -> None:
        self._candidates = list(candidates)

    def generate(self, documents: Sequence[str]) -> List[Candidate]:
        _ = documents
        return list(self._candidates)


def test_label_cluster_returns_ranked_phrases() -> None:
    """End-to-end smoke test that verifies ranking and scoring."""

    documents = ["doc a", "doc b", "doc c"]
    mapping = {
        "doc a": np.array([1.0, 0.0, 0.0], dtype=float),
        "doc b": np.array([0.9, 0.1, 0.0], dtype=float),
        "doc c": np.array([1.1, 0.1, 0.0], dtype=float),
        "green energy": np.array([0.95, 0.05, 0.0], dtype=float),
        "renewable power": np.array([0.92, 0.08, 0.0], dtype=float),
        "fossil fuel": np.array([0.2, 0.8, 0.0], dtype=float),
    }
    embedder = FakeEmbedder(mapping)
    candidates = [
        Candidate(text="green energy", term_frequency=5, document_frequency=3),
        Candidate(text="renewable power", term_frequency=3, document_frequency=2),
        Candidate(text="fossil fuel", term_frequency=4, document_frequency=2),
    ]
    candidate_generator = FakeCandidateGenerator(candidates)

    result = label_cluster(
        documents,
        embedder=embedder,
        candidate_generator=candidate_generator,
        top_k=2,
        diversity=0.6,
    )

    assert result["top_phrases"][0] == "green energy"
    assert set(result["top_phrases"]) <= {"green energy", "renewable power", "fossil fuel"}
    assert result["best_phrase"] == "green energy"
    assert "cohesion" in result["scores"]
    assert 0.0 <= result["scores"]["cohesion"] <= 1.0


def test_label_cluster_deduplicates_similar_candidates() -> None:
    """Highly similar candidates should collapse to a single entry."""

    documents = ["doc a", "doc b"]
    mapping = {
        "doc a": np.array([1.0, 0.0, 0.0], dtype=float),
        "doc b": np.array([1.0, 0.0, 0.0], dtype=float),
        "machine learning": np.array([0.7, 0.3, 0.0], dtype=float),
        "machine intelligence": np.array([0.699, 0.301, 0.0], dtype=float),
    }
    embedder = FakeEmbedder(mapping)
    candidates = [
        Candidate(text="machine learning", term_frequency=4, document_frequency=2),
        Candidate(text="machine intelligence", term_frequency=3, document_frequency=2),
    ]
    candidate_generator = FakeCandidateGenerator(candidates)

    result = label_cluster(
        documents,
        embedder=embedder,
        candidate_generator=candidate_generator,
        top_k=5,
    )

    assert result["top_phrases"] == ["machine learning"]
    assert result["best_phrase"] == "machine learning"

