"""Tests for the publication clustering helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy
import pandas  # type: ignore[import-untyped]
import pytest
from sklearn.cluster import KMeans  # type: ignore[import-untyped]

sys.path.append(str(Path(__file__).resolve().parents[1]))

from _scripts import build_json


@pytest.fixture(scope="session")
def fixture_records() -> list[dict[str, object]]:
    """Load the synthetic publication dataset for clustering tests."""

    fixture_path = Path(__file__).parent / "fixtures" / "cluster_fixture.json"
    return json.loads(fixture_path.read_text())


def test_limit_citations_prioritises_recent_records() -> None:
    """Recent publications should be preferred when limiting the dataset."""

    citations = pandas.DataFrame(
        [
            {
                "author_pub_id": "old-high",
                "pub_year": 2018,
                "num_citations": 75,
            },
            {
                "author_pub_id": "recent-low",
                "pub_year": 2024,
                "num_citations": 2,
            },
            {
                "author_pub_id": "recent-high",
                "pub_year": 2024,
                "num_citations": 11,
            },
            {
                "author_pub_id": "mid",
                "pub_year": 2022,
                "num_citations": 5,
            },
        ]
    )

    limited = build_json.limit_citations(citations, max_records=2)

    assert limited["author_pub_id"].tolist() == ["recent-high", "recent-low"]


def test_limit_citations_rejects_non_positive_limits() -> None:
    """The limiter should validate the configured maximum number of records."""

    citations = pandas.DataFrame(
        [
            {
                "author_pub_id": "only",
                "pub_year": 2020,
                "num_citations": 1,
            }
        ]
    )

    with pytest.raises(ValueError):
        build_json.limit_citations(citations, max_records=0)


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


def test_cluster_points_refines_using_projection() -> None:
    """Clusters should be reinitialised in the projection space."""

    rng = numpy.random.default_rng(17)
    n_clusters = build_json.DEFAULT_KMEANS_CLUSTERS
    points_per_cluster = 12
    feature_dim = 64

    embedding_groups: list[numpy.ndarray] = []
    projection_groups: list[numpy.ndarray] = []
    for cluster_id in range(n_clusters):
        angle = (2 * numpy.pi * cluster_id) / float(n_clusters)
        center_2d = numpy.array([numpy.cos(angle) * 5.0, numpy.sin(angle) * 5.0])
        embedding_cluster = rng.normal(
            loc=0.0,
            scale=0.3,
            size=(points_per_cluster, feature_dim),
        )
        embedding_cluster[:, :2] += center_2d
        projection_cluster = rng.normal(
            loc=center_2d,
            scale=0.2,
            size=(points_per_cluster, 2),
        )
        embedding_groups.append(embedding_cluster)
        projection_groups.append(projection_cluster)

    embeddings = numpy.vstack(embedding_groups)
    projection = numpy.vstack(projection_groups)

    result = build_json.cluster_points_from_embeddings(
        embeddings,
        projection,
        random_state=11,
    )

    assert result.space == "tsne(init=pca50)"

    reduced = build_json.reduce_for_clustering(embeddings, random_state=11)
    base_clusterer = KMeans(n_clusters=n_clusters, random_state=11)
    base_labels = base_clusterer.fit_predict(reduced.matrix)

    centroids = numpy.vstack(
        [projection[base_labels == cluster_id].mean(axis=0) for cluster_id in range(n_clusters)]
    )
    expected_clusterer = KMeans(
        n_clusters=n_clusters,
        init=centroids,
        n_init=1,
        random_state=11,
    )
    expected_labels = expected_clusterer.fit_predict(projection)

    numpy.testing.assert_array_equal(result.labels, expected_labels)


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


def test_summarize_clusters_uses_ctfidf_labels(
    fixture_records: List[Dict[str, object]],
) -> None:
    """Summaries should expose c-TF-IDF labels when they are available."""

    citations = pandas.DataFrame(fixture_records)
    embeddings = numpy.array([[rec["x"], rec["y"]] for rec in fixture_records])
    labels = numpy.array([0, 0, 0, 0, 1, 1, 1, 1, -1])

    citations["x"] = embeddings[:, 0]
    citations["y"] = embeddings[:, 1]

    ctfidf_map = {0: "design automation"}
    summaries = build_json.summarize_clusters(
        citations, labels.tolist(), ctfidf=ctfidf_map
    )

    lookup = {summary.cluster_id: summary for summary in summaries}

    assert lookup[0].label == "design automation"
    assert lookup[1].label == ""


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
