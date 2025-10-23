"""Tests for the publication clustering helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy
import pandas  # type: ignore[import-untyped]
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from _scripts import build_json


class DummyLabelModel:
    """Minimal KeyBERT-like model returning deterministic labels."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def extract_keywords(self, text: str, **kwargs: Any) -> List[tuple[str, float]]:
        self.calls.append({"text": text, "kwargs": kwargs})
        return [("design automation", 0.9), ("robot teamwork", 0.8)]


@pytest.fixture(scope="session")
def fixture_records() -> list[dict[str, object]]:
    """Load the synthetic publication dataset for clustering tests."""

    fixture_path = Path(__file__).parent / "fixtures" / "cluster_fixture.json"
    return json.loads(fixture_path.read_text())


def test_compute_projection_is_deterministic() -> None:
    """t-SNE + PCA should return repeatable coordinates for a fixed seed."""

    rng = numpy.random.default_rng(0)
    embeddings = rng.normal(size=(10, 64))

    result_one = build_json.compute_projection(embeddings, random_state=7)
    result_two = build_json.compute_projection(embeddings, random_state=7)

    numpy.testing.assert_allclose(result_one.coordinates, result_two.coordinates)
    assert result_one.perplexity == 3


def test_cluster_points_from_embeddings_forms_clusters() -> None:
    """K-means should partition the embeddings into the configured groups."""

    rng = numpy.random.default_rng(42)
    cluster_a = rng.normal(loc=0.0, scale=0.2, size=(20, 64))
    cluster_b = rng.normal(loc=5.0, scale=0.2, size=(20, 64))
    embeddings = numpy.vstack([cluster_a, cluster_b])

    result = build_json.cluster_points_from_embeddings(embeddings, random_state=13)

    unique_labels = set(result.labels)
    assert len(unique_labels) == build_json.DEFAULT_KMEANS_CLUSTERS
    assert result.algorithm == "kmeans"


def test_cluster_points_from_embeddings_finds_multiple_topics() -> None:
    """The clustering heuristics should surface several granular topics."""

    rng = numpy.random.default_rng(3)
    points_per_cluster = 40
    clusters: list[numpy.ndarray] = []
    for index in range(7):
        angle = 2 * numpy.pi * index / 7
        center = numpy.array([
            numpy.cos(angle) * 6.0,
            numpy.sin(angle) * 6.0,
        ])
        samples = rng.normal(loc=0.0, scale=0.35, size=(points_per_cluster, 6))
        samples[:, :2] += center
        clusters.append(samples)

    embeddings = numpy.vstack(clusters)

    result = build_json.cluster_points_from_embeddings(embeddings, random_state=11)

    unique_labels = set(result.labels)
    assert len(unique_labels) == build_json.DEFAULT_KMEANS_CLUSTERS


def test_ctfidf_labels_produces_multiword_phrases() -> None:
    """The c-TF-IDF helper should surface informative phrases per cluster."""

    citations = pandas.DataFrame(
        [
            {
                "author_pub_id": "a1",
                "bib_dict": {
                    "title": "Robot teamwork coordination",
                    "abstract": "Robot teamwork improves.",
                },
            },
            {
                "author_pub_id": "a2",
                "bib_dict": {
                    "title": "Robot teamwork efficiency",
                    "abstract": "Robot teamwork analysis.",
                },
            },
            {
                "author_pub_id": "b1",
                "bib_dict": {
                    "title": "Design automation for manufacturing",
                    "abstract": "Automation design study.",
                },
            },
            {
                "author_pub_id": "b2",
                "bib_dict": {
                    "title": "Design automation workflow",
                    "abstract": "Automation pipeline details.",
                },
            },
        ]
    )
    labels = numpy.array([0, 0, 1, 1])

    phrases = build_json.ctfidf_labels(citations, labels.tolist(), top_k=2)

    assert phrases[0].startswith("robot teamwork")
    assert "design automation" in phrases[1]


def test_summarize_clusters_uses_ctfidf_and_fallback(
    fixture_records: List[Dict[str, object]],
) -> None:
    """Summaries should prefer c-TF-IDF labels while falling back to KeyBERT."""

    citations = pandas.DataFrame(fixture_records)
    embeddings = numpy.array([[rec["x"], rec["y"]] for rec in fixture_records])
    labels = numpy.array([0, 0, 0, 0, 1, 1, 1, 1, -1])

    citations["x"] = embeddings[:, 0]
    citations["y"] = embeddings[:, 1]
    dummy_model = DummyLabelModel()

    ctfidf_map = {0: "design automation", 1: ""}
    summaries = build_json.summarize_clusters(
        citations, labels.tolist(), dummy_model, ctfidf=ctfidf_map
    )

    lookup = {summary.cluster_id: summary for summary in summaries}

    assert lookup[0].label == "design automation"
    assert lookup[1].label == "design automation, robot teamwork"
    assert (
        dummy_model.calls
    ), "Fallback labeller should be invoked for missing c-TF-IDF labels."


def test_dump_payload_skips_identical_payload(tmp_path: Path) -> None:
    """dump_payload should avoid rewriting unchanged JSON content."""

    payload: Dict[str, object] = {
        "records": [],
        "clusters": [],
        "meta": {"random_state": 0},
    }
    workspace = tmp_path / "repo"
    (workspace / "assets/json").mkdir(parents=True)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))

    try:
        first_write = build_json.dump_payload(payload, force=False)
        second_write = build_json.dump_payload(payload, force=False)
    finally:
        monkeypatch.undo()

    target = workspace / "assets/json/pubs.json"
    assert first_write is True
    assert second_write is False
    assert json.loads(target.read_text(encoding="utf-8")) == payload


def test_build_cluster_text_deduplicates_entries(
    fixture_records: List[Dict[str, object]],
) -> None:
    """Cluster text should drop duplicate fragments and respect the char cap."""

    citations = pandas.DataFrame(fixture_records[:2])
    citations.loc[0, "bib_dict"]["abstract"] = "Repeated abstract"
    citations.loc[1, "bib_dict"]["abstract"] = "Repeated abstract"

    text = build_json.build_cluster_text(citations, max_chars=50)

    assert text.count("Repeated abstract") == 1
