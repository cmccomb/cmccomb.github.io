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
from _scripts.topic_labels import (
    CURATED_LABELS,
    curated_label,
    normalise_topic_label,
    suitable_label,
)


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

    phrases = build_json.ctfidf_labels(citations, labels.tolist())

    assert phrases[0].startswith("robot teamwork")
    assert "design automation" in phrases[1]


def test_ctfidf_labels_prioritises_cluster_wide_phrases() -> None:
    """A phrase from a few papers should not name a much broader cluster."""

    citations = pandas.DataFrame(
        [
            {
                "author_pub_id": "a1",
                "bib_dict": {
                    "title": "Virtual face-to-face collaboration",
                    "abstract": "Face-to-face collaboration face-to-face collaboration.",
                },
            },
            {
                "author_pub_id": "a2",
                "bib_dict": {
                    "title": "Face-to-face team collaboration",
                    "abstract": "Face-to-face collaboration face-to-face collaboration.",
                },
            },
            {
                "author_pub_id": "a3",
                "bib_dict": {
                    "title": "Design teams coordinate concepts",
                    "abstract": "Design teams compare concepts.",
                },
            },
            {
                "author_pub_id": "a4",
                "bib_dict": {
                    "title": "Design teams select ideas",
                    "abstract": "Design teams select concepts.",
                },
            },
            {
                "author_pub_id": "a5",
                "bib_dict": {
                    "title": "Design teams improve collaboration",
                    "abstract": "Design teams study collaboration.",
                },
            },
            {
                "author_pub_id": "b1",
                "bib_dict": {
                    "title": "Manufacturing systems optimisation",
                    "abstract": "Manufacturing systems improve production.",
                },
            },
            {
                "author_pub_id": "b2",
                "bib_dict": {
                    "title": "Manufacturing systems reliability",
                    "abstract": "Manufacturing systems support production.",
                },
            },
        ]
    )
    labels = numpy.array([0, 0, 0, 0, 0, 1, 1])

    phrases = build_json.ctfidf_labels(citations, labels.tolist())

    assert phrases[0] == "design teams"


def test_reviewed_topics_apply_only_to_exact_membership() -> None:
    """Reviewed names outrank generated phrases until cluster membership changes."""

    memberships = {label: ids for ids, label in CURATED_LABELS.items()}
    rows = [
        {"author_pub_id": publication_id, "bib_dict": {"title": title}}
        for label, title in (
            ("design methods", "Lattice manufacturing systems"),
            ("design decisions", "Robot teamwork process"),
        )
        for publication_id in sorted(memberships[label])
    ]
    labels = (
        [0] * len(memberships["design methods"])
        + [1] * len(memberships["design decisions"])
    )
    rows.extend([
        {"author_pub_id": "other-1", "bib_dict": {"title": "Coastal erosion study"}},
        {"author_pub_id": "other-2", "bib_dict": {"title": "Coastal erosion data"}},
    ])
    labels.extend([2, 2])

    assert build_json.ctfidf_labels(pandas.DataFrame(rows), labels) == {
        0: "design methods",
        1: "design decisions",
        2: "coastal erosion",
    }
    assert curated_label([*memberships["design methods"], "new-publication"]) is None
    assert not suitable_label("market in")
    assert not suitable_label("work we")


def test_ctfidf_labels_resolves_duplicate_reviewed_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even conflicting reviewed names should fall back to distinct phrases."""

    citations = pandas.DataFrame([
        {"author_pub_id": "a", "bib_dict": {"title": "Lattice design methods"}},
        {"author_pub_id": "b", "bib_dict": {"title": "Manufacturing processes"}},
    ])
    monkeypatch.setattr(
        build_json,
        "curated_label",
        lambda publication_ids: (
            "Additive-Manufacturing"
            if "a" in set(publication_ids)
            else "additive manufacturing"
        ),
    )

    generated = build_json.ctfidf_labels(citations, [0, 1])
    normalised = [normalise_topic_label(label) for label in generated.values()]
    assert len(set(normalised)) == 2
    assert normalised.count("additive manufacturing") == 1


def test_unique_topic_assignment_chooses_supported_alternative() -> None:
    """A contested first choice gives way to the best available distinct name."""

    chosen = build_json._assign_unique_topic_labels(
        {
            3: ["language models", "design methods"],
            11: ["language models", "large language models"],
        },
        {3: "language models"},
    )

    assert chosen == {3: "language models", 11: "large language models"}


def test_unique_topic_assignment_avoids_cascading_collisions() -> None:
    """Assigning one shared phrase should account for other clusters' choices."""

    chosen = build_json._assign_unique_topic_labels(
        {
            0: ["shared topic", "first alternative"],
            1: ["shared topic", "second choice"],
            2: ["second choice", "third choice"],
        },
        {},
    )

    assert chosen == {
        0: "first alternative",
        1: "shared topic",
        2: "second choice",
    }


def test_unique_topic_assignment_fails_without_distinct_candidates() -> None:
    """The build still fails if the source cannot support unique topic names."""

    with pytest.raises(ValueError, match="No distinct supported topic labels"):
        build_json._assign_unique_topic_labels(
            {3: ["language models"], 11: ["language models"]}, {}
        )


def test_summarize_clusters_uses_ctfidf_labels(
    fixture_records: List[Dict[str, object]],
) -> None:
    """Summaries should use the supplied c-TF-IDF labels."""

    citations = pandas.DataFrame(fixture_records)
    embeddings = numpy.array([[rec["x"], rec["y"]] for rec in fixture_records])
    labels = numpy.array([0, 0, 0, 0, 1, 1, 1, 1, -1])

    citations["x"] = embeddings[:, 0]
    citations["y"] = embeddings[:, 1]
    ctfidf_map = {0: "design automation", 1: "robot teamwork"}
    summaries = build_json.summarize_clusters(
        citations, labels.tolist(), ctfidf=ctfidf_map
    )

    lookup = {summary.cluster_id: summary for summary in summaries}

    assert lookup[0].label == "design automation"
    assert lookup[1].label == "robot teamwork"


def test_missing_publication_year_uses_current_year() -> None:
    """Missing publication years should not stay pinned to an old calendar year."""

    assert build_json._extract_pub_year({}) == build_json.datetime.now(
        build_json.timezone.utc
    ).year


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


def test_build_payload_preserves_optional_paper_links(monkeypatch: pytest.MonkeyPatch) -> None:
    """A routine dataset refresh must retain available paper access metadata."""
    frame = pandas.DataFrame([
        {"embedding": [1.0, 0.0], "author_pub_id": "a:one", "pub_year": 2026,
         "num_citations": 1, "bib_dict": {"title": "First paper"},
         "pub_url": "https://doi.org/10.1234/example", "eprint_url": None},
        {"embedding": [0.0, 1.0], "author_pub_id": "a:two", "pub_year": 2026,
         "num_citations": 0, "bib_dict": {"title": "Second paper"},
         "pub_url": None, "eprint_url": "https://arxiv.org/abs/2601.12345"},
    ])
    monkeypatch.setattr(build_json, "compute_projection", lambda *args, **kwargs:
                        build_json.ProjectionResult(numpy.array([[0.0, 0.0], [1.0, 1.0]]), 1))
    monkeypatch.setattr(build_json, "cluster_points_from_embeddings", lambda *args, **kwargs:
                        build_json.ClusteringResult(numpy.array([0, 0]), "test", "test", 1))
    monkeypatch.setattr(build_json, "ctfidf_labels", lambda *args, **kwargs: {})
    monkeypatch.setattr(build_json, "summarize_clusters", lambda *args, **kwargs: [])
    payload = build_json.build_payload(frame)
    records = payload["records"]
    assert records[0]["pub_url"] == "https://doi.org/10.1234/example"
    assert "eprint_url" not in records[0]
    assert records[1]["eprint_url"] == "https://arxiv.org/abs/2601.12345"
    assert "pub_url" not in records[1]
    json.dumps(payload, allow_nan=False)
    without_links = build_json.build_payload(frame.drop(columns=["pub_url", "eprint_url"]))
    assert len(without_links["records"]) == 2
