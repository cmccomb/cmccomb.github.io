"""Tests for the publication clustering helpers."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy
import pandas  # type: ignore[import-untyped]
import pytest
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score

sys.path.append(str(Path(__file__).resolve().parents[1]))

from _scripts import build_json
from _scripts.derive_topic_labels import complete_terms


@pytest.fixture(scope="session")
def fixture_records() -> list[dict[str, object]]:
    """Load the synthetic publication dataset for clustering tests."""

    fixture_path = Path(__file__).parent / "fixtures" / "cluster_fixture.json"
    return json.loads(fixture_path.read_text())


def test_compute_projection_is_deterministic() -> None:
    """A fixed input and seed should produce repeatable finite map coordinates."""

    rng = numpy.random.default_rng(0)
    embeddings = rng.normal(size=(10, 64))

    result_one = build_json.compute_projection(embeddings, random_state=7)
    result_two = build_json.compute_projection(embeddings, random_state=7)

    numpy.testing.assert_allclose(result_one.coordinates, result_two.coordinates)
    assert result_one.perplexity == 3
    assert numpy.isfinite(result_one.coordinates).all()


def test_semantic_clustering_recovers_separated_topics_across_seeds() -> None:
    """Clearly separated semantic topics should survive different fit seeds."""

    rng = numpy.random.default_rng(42)
    n_clusters = build_json.DEFAULT_KMEANS_CLUSTERS
    points_per_cluster = 8
    embeddings = numpy.vstack([
        numpy.eye(64)[cluster] * 10
        + rng.normal(scale=.15, size=(points_per_cluster, 64))
        for cluster in range(n_clusters)
    ])
    planted = numpy.repeat(numpy.arange(n_clusters), points_per_cluster)

    first = build_json.cluster_points_from_embeddings(embeddings, random_state=13)
    second = build_json.cluster_points_from_embeddings(embeddings, random_state=84)

    assert first.algorithm == "kmeans"
    assert len(numpy.unique(first.labels)) == n_clusters
    assert adjusted_rand_score(planted, first.labels) > .95
    assert adjusted_rand_score(first.labels, second.labels) > .95


def test_projection_preserves_clear_semantic_neighborhoods() -> None:
    """An easily separable cosine geometry should retain its local neighbors."""

    rng = numpy.random.default_rng(17)
    embeddings = numpy.vstack([
        numpy.eye(16)[cluster] * 8
        + rng.normal(scale=.3, size=(20, 16))
        for cluster in range(3)
    ])

    result = build_json.compute_projection(embeddings, random_state=42)

    assert result.coordinates.shape == (len(embeddings), 2)
    assert trustworthiness(
        embeddings, result.coordinates, n_neighbors=5, metric="cosine"
    ) > .9


def test_build_payload_keeps_nonoversized_topics_independent_of_map_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Projection changes must leave topics already under the size cap intact."""

    rng = numpy.random.default_rng(29)
    n_clusters = build_json.DEFAULT_KMEANS_CLUSTERS
    points_per_cluster = 3
    embeddings = numpy.vstack([
        numpy.eye(64)[cluster] * 10
        + rng.normal(scale=.1, size=(points_per_cluster, 64))
        for cluster in range(n_clusters)
    ])
    frame = pandas.DataFrame([
        {"embedding": vector.tolist(), "author_pub_id": f"paper-{index}",
         "pub_year": 2026, "num_citations": 1,
         "bib_dict": {"title": f"Synthetic publication {index}"}}
        for index, vector in enumerate(embeddings)
    ])
    first_layout = numpy.column_stack([
        numpy.repeat(numpy.arange(n_clusters), points_per_cluster),
        numpy.tile(numpy.arange(points_per_cluster), n_clusters),
    ]).astype(float)
    second_layout = rng.permutation(first_layout) * numpy.array([-3., 2.])
    monkeypatch.setattr(build_json, "ctfidf_labels", lambda citations, labels: {
        int(group): f"research topic {group}" for group in set(labels)
    })
    monkeypatch.setattr(build_json, "compute_projection", lambda *args, **kwargs:
                        build_json.ProjectionResult(first_layout, 3))
    first = build_json.build_payload(frame, random_state=42)
    monkeypatch.setattr(build_json, "compute_projection", lambda *args, **kwargs:
                        build_json.ProjectionResult(second_layout, 3))
    second = build_json.build_payload(frame, random_state=42)

    assert {
        record["author_pub_id"]: record["cluster_id"] for record in first["records"]
    } == {
        record["author_pub_id"]: record["cluster_id"] for record in second["records"]
    }
    assert [(row["x"], row["y"]) for row in first["records"]] != [
        (row["x"], row["y"]) for row in second["records"]
    ]
    assert "tsne" not in first["meta"]["clustering"]["space"]
    assert max(numpy.unique(
        [row["cluster_id"] for row in first["records"]], return_counts=True
    )[1]) <= len(frame) // 8
    assert first["meta"]["clustering"]["projection_refinement"] is False
    assert first["meta"]["clustering"]["n_init"] == build_json.DEFAULT_KMEANS_N_INIT


def test_size_cap_splits_only_oversized_parent_topics() -> None:
    """Capped children retain their semantic parent without merging other topics."""

    rng = numpy.random.default_rng(19)
    parents = numpy.repeat(numpy.arange(7), [25, 18, 9, 8, 7, 6, 7])
    points = numpy.eye(8)[parents] * 10 + rng.normal(scale=.5, size=(80, 8))
    projection = rng.normal(size=(80, 2))

    labels = build_json.cap_cluster_sizes(points, parents, projection, random_state=42)

    ids, counts = numpy.unique(labels, return_counts=True)
    numpy.testing.assert_array_equal(ids, numpy.arange(len(ids)))
    assert len(labels) == len(parents)
    assert 0 < counts.min() <= counts.max() <= len(parents) // 8
    for child in ids:
        assert len(numpy.unique(parents[labels == child])) == 1
    assert len(numpy.unique(labels[parents == 0])) == 3
    assert len(numpy.unique(labels[parents == 1])) == 2
    for parent in range(2, 7):
        assert len(numpy.unique(labels[parents == parent])) == 1


def test_size_cap_handles_threshold_and_repeats_deterministically() -> None:
    """A topic exactly at the limit stays whole while larger ones split reliably."""

    rng = numpy.random.default_rng(23)
    parents = numpy.repeat([10, 30, 50], [8, 9, 47])
    points = rng.normal(size=(64, 6))
    projection = rng.normal(size=(64, 2))

    first = build_json.cap_cluster_sizes(points, parents, projection, random_state=42)
    second = build_json.cap_cluster_sizes(points, parents, projection, random_state=42)

    numpy.testing.assert_array_equal(first, second)
    ids, counts = numpy.unique(first, return_counts=True)
    numpy.testing.assert_array_equal(ids, numpy.arange(len(ids)))
    assert counts.min() > 0 and counts.max() <= 8
    assert len(numpy.unique(first[parents == 10])) == 1
    assert len(numpy.unique(first[parents == 30])) == 2
    assert len(numpy.unique(first[parents == 50])) == 6


def test_size_cap_supports_collections_smaller_than_eight_papers() -> None:
    """The minimum cap of one still assigns every paper to a nonempty topic."""

    points = numpy.random.default_rng(31).normal(size=(5, 3))

    labels = build_json.cap_cluster_sizes(points, numpy.zeros(5, dtype=int), random_state=42)

    ids, counts = numpy.unique(labels, return_counts=True)
    numpy.testing.assert_array_equal(ids, numpy.arange(5))
    numpy.testing.assert_array_equal(counts, numpy.ones(5, dtype=int))


def test_size_cap_does_not_peel_off_a_single_outlier_child() -> None:
    """A capped split should retain usable child sizes around a strong outlier."""

    parents = numpy.repeat(numpy.arange(8), [11, 10, 10, 10, 10, 10, 10, 9])
    points = numpy.eye(8)[parents] * 10
    points[10] = numpy.ones(8) * 100

    labels = build_json.cap_cluster_sizes(points, parents, random_state=42)

    children, counts = numpy.unique(labels[parents == 0], return_counts=True)
    assert len(children) == 2
    assert counts.min() >= 2
    assert counts.max() <= 10


def test_oversized_semantically_tied_topic_respects_separate_visual_regions() -> None:
    """When semantics tie, local splitting should retain clear display islands."""

    rng = numpy.random.default_rng(37)
    parents = numpy.repeat(numpy.arange(7), [20, 10, 10, 10, 10, 10, 10])
    points = numpy.eye(8)[parents] * 10
    projection = rng.normal(scale=.1, size=(80, 2))
    projection[:10, 0] -= 100
    projection[10:20, 0] += 100

    labels = build_json.cap_cluster_sizes(points, parents, projection, random_state=42)

    assert len(numpy.unique(labels[:10])) == 1
    assert len(numpy.unique(labels[10:20])) == 1
    assert labels[0] != labels[10]
    assert numpy.unique(labels, return_counts=True)[1].max() <= 10
    for parent in range(1, 7):
        assert len(numpy.unique(labels[parents == parent])) == 1


def test_proximity_splits_a_small_semantically_distinct_visual_island() -> None:
    """A group below the size cap can still have two meaningful map islands."""

    rng = numpy.random.default_rng(107)
    embeddings = numpy.vstack([
        numpy.eye(8)[0] * 8 + rng.normal(scale=.08, size=(5, 8)),
        numpy.eye(8)[1] * 8 + rng.normal(scale=.08, size=(5, 8)),
        numpy.eye(8)[2] * 8 + rng.normal(scale=.08, size=(10, 8)),
    ])
    projection = numpy.vstack([
        numpy.array([-10, 0]) + rng.normal(scale=.08, size=(5, 2)),
        numpy.array([10, 0]) + rng.normal(scale=.08, size=(5, 2)),
        numpy.array([0, 10]) + rng.normal(scale=.08, size=(10, 2)),
    ])
    parents = numpy.repeat([0, 1], [10, 10])

    first, count = build_json.split_visually_dispersed_topics(
        embeddings, embeddings, parents, projection, random_state=42,
    )
    second, repeated_count = build_json.split_visually_dispersed_topics(
        embeddings, embeddings, parents, projection, random_state=42,
    )

    assert count == repeated_count == 1
    numpy.testing.assert_array_equal(first, second)
    assert len(numpy.unique(first[:10])) == 2
    assert len(numpy.unique(first[10:])) == 1
    assert first[0] != first[5]


def test_proximity_does_not_split_a_visual_artifact_without_semantic_support() -> None:
    """Separated t-SNE positions alone cannot redefine a coherent topic."""

    rng = numpy.random.default_rng(109)
    embeddings = numpy.vstack([
        numpy.eye(8)[0] * 8 + rng.normal(scale=.08, size=(10, 8)),
        numpy.eye(8)[1] * 8 + rng.normal(scale=.08, size=(10, 8)),
    ])
    projection = numpy.vstack([
        numpy.array([-10, 0]) + rng.normal(scale=.08, size=(5, 2)),
        numpy.array([10, 0]) + rng.normal(scale=.08, size=(5, 2)),
        numpy.array([0, 10]) + rng.normal(scale=.08, size=(10, 2)),
    ])
    parents = numpy.repeat([0, 1], [10, 10])

    result, count = build_json.split_visually_dispersed_topics(
        embeddings, embeddings, parents, projection, random_state=42,
    )

    assert count == 0
    numpy.testing.assert_array_equal(result, parents)


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


def test_ctfidf_labels_keeps_complete_hyphenated_model_terms() -> None:
    """Singular and plural evidence should support a complete technical term."""

    citations = pandas.DataFrame([
        {"author_pub_id": "a1", "bib_dict": {
            "title": "Vision-language model benchmarks",
            "abstract": "A vision-language model supports simulation analysis.",
        }},
        {"author_pub_id": "a2", "bib_dict": {
            "title": "Vision-language models assess images",
            "abstract": "Vision-language models solve visual tasks.",
        }},
        {"author_pub_id": "b1", "bib_dict": {"title": "Coastal erosion measurements"}},
        {"author_pub_id": "b2", "bib_dict": {"title": "Coastal erosion assessment"}},
    ])

    generated = build_json.ctfidf_labels(citations, [0, 0, 1, 1])

    assert generated[0] in {"vision-language model", "vision-language models"}


def test_ctfidf_labels_limits_each_papers_influence() -> None:
    """Repeated abstract wording should not outweigh a theme shared by papers."""

    citations = pandas.DataFrame([
        {"author_pub_id": "a1", "bib_dict": {
            "title": "Measurement workflow for visual studies",
            "abstract": "Niche sensor experiments. " * 200,
        }},
        {"author_pub_id": "a2", "bib_dict": {
            "title": "Calibration protocol for optical analysis",
            "abstract": "Niche sensor experiments.",
        }},
        {"author_pub_id": "a3", "bib_dict": {
            "title": "Design teams coordinate concepts",
            "abstract": "Design teams compare choices.",
        }},
        {"author_pub_id": "a4", "bib_dict": {
            "title": "Design teams select ideas",
            "abstract": "Design teams refine layouts.",
        }},
        {"author_pub_id": "a5", "bib_dict": {
            "title": "Design teams improve collaboration",
            "abstract": "Design teams review alternatives.",
        }},
        {"author_pub_id": "b1", "bib_dict": {"title": "Coastal erosion measurements"}},
        {"author_pub_id": "b2", "bib_dict": {"title": "Coastal erosion assessment"}},
    ])

    generated = build_json.ctfidf_labels(citations, [0, 0, 0, 0, 0, 1, 1])

    assert generated[0] == "design teams"


def test_ctfidf_labels_preserves_standalone_hyphenated_topics() -> None:
    """A complete compound noun remains eligible without additional words."""

    citations = pandas.DataFrame([
        {"author_pub_id": "a1", "bib_dict": {"title": "Super-resolution methods for flow"}},
        {"author_pub_id": "a2", "bib_dict": {"title": "Super-resolution evaluation improves fields"}},
        {"author_pub_id": "b1", "bib_dict": {"title": "Coastal erosion measurements"}},
        {"author_pub_id": "b2", "bib_dict": {"title": "Coastal erosion assessment"}},
    ])

    assert build_json.ctfidf_labels(citations, [0, 0, 1, 1])[0] == "super-resolution"


def test_phrase_completion_does_not_cascade_into_a_narrower_subtopic() -> None:
    """Successive extensions must not erase the well-supported intermediate term."""

    retained = complete_terms({
        "alpha beta": set(range(10)),
        "alpha beta gamma": set(range(8)),
        "alpha beta gamma delta": set(range(7)),
        "alpha beta gamma delta epsilon": set(range(6)),
    })

    assert "alpha beta gamma" in retained


def test_ctfidf_labels_are_independent_of_record_order() -> None:
    """Reordering unchanged papers should not rename their topics."""

    citations = pandas.DataFrame([
        {"author_pub_id": "a1", "bib_dict": {"title": "Vision-language model benchmarks"}},
        {"author_pub_id": "a2", "bib_dict": {"title": "Vision-language models assess images"}},
        {"author_pub_id": "b1", "bib_dict": {"title": "Coastal erosion measurements"}},
        {"author_pub_id": "b2", "bib_dict": {"title": "Coastal erosion assessment"}},
    ])
    labels = [0, 0, 1, 1]
    order = [3, 1, 0, 2]

    assert build_json.ctfidf_labels(citations, labels) == build_json.ctfidf_labels(
        citations.iloc[order].reset_index(drop=True), [labels[index] for index in order]
    )


def test_unique_topic_assignment_chooses_supported_alternative() -> None:
    """A contested first choice gives way to the best available distinct name."""

    chosen = build_json._assign_unique_topic_labels(
        {
            3: [("language model", 9.0), ("design methods", 8.0)],
            11: [("language models", 10.0), ("large language models", 1.0)],
        },
    )

    assert chosen == {3: "design methods", 11: "language models"}


def test_unique_topic_assignment_avoids_cascading_collisions() -> None:
    """Assigning one shared phrase should account for other clusters' choices."""

    chosen = build_json._assign_unique_topic_labels(
        {
            0: [("shared topic", 10.0), ("first alternative", 9.0)],
            1: [("shared topic", 10.0), ("second choice", 5.0)],
            2: [("second choice", 10.0), ("third choice", 1.0)],
        },
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
            {3: [("language models", 1.0)], 11: [("language models", 1.0)]}
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


def test_relabel_payload_preserves_snapshot_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relabeling changes topic names and their provenance without rebuilding data."""

    payload = {
        "meta": {
            "built_at_utc": "2026-10-01T12:00:00+00:00",
            "dataset_commit": "a" * 40,
            "projection": {"method": "tsne+pca", "perplexity": 30},
            "labeling": {"method": "old"},
        },
        "records": [
            {"author_pub_id": "a1", "cluster_id": 0, "x": 1.25, "y": -2.5,
             "pub_year": 2024, "num_citations": 17,
             "bib_dict": {"title": "Robot teamwork coordination", "abstract": "Source abstract."},
             "pub_url": "https://example.com/paper"},
            {"author_pub_id": "b1", "cluster_id": 1, "x": 3.0, "y": 4.0,
             "pub_year": 2025, "num_citations": 3,
             "bib_dict": {"title": "Coastal erosion measurements"}},
        ],
        "clusters": [
            {"id": 0, "label": "old robot topic", "centroid": {"x": 1.25, "y": -2.5}},
            {"id": 1, "label": "old coastal topic", "centroid": {"x": 3.0, "y": 4.0}},
        ],
    }
    original = copy.deepcopy(payload)
    monkeypatch.setattr(
        build_json, "derive_labels",
        lambda citations, labels: {0: "robot teamwork", 1: "coastal erosion"},
    )

    relabeled = build_json.relabel_payload(payload)

    assert payload == original
    assert relabeled["records"] == original["records"]
    assert {
        key: value for key, value in relabeled["meta"].items() if key != "labeling"
    } == {
        key: value for key, value in original["meta"].items() if key != "labeling"
    }
    assert relabeled["meta"]["labeling"] != original["meta"]["labeling"]
    expected_clusters = copy.deepcopy(original["clusters"])
    expected_clusters[0]["label"] = "robot teamwork"
    expected_clusters[1]["label"] = "coastal erosion"
    assert relabeled["clusters"] == expected_clusters
    relabeled["records"][0]["bib_dict"]["title"] = "Changed result"
    assert payload == original


def test_committed_topic_names_are_automatically_reproducible() -> None:
    """Published titles must come from the generator rather than snapshot edits."""

    snapshot_path = Path(__file__).resolve().parents[1] / "assets/json/pubs.json"
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))

    regenerated = build_json.relabel_payload(payload)

    assert {
        cluster["id"]: cluster["label"] for cluster in payload["clusters"]
    } == {
        cluster["id"]: cluster["label"] for cluster in regenerated["clusters"]
    }, "Regenerate the snapshot's topic names with the automatic labeling command."


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
    assert payload["meta"]["quality"]["projection_trustworthiness"] is None
    records = payload["records"]
    assert records[0]["pub_url"] == "https://doi.org/10.1234/example"
    assert "eprint_url" not in records[0]
    assert records[1]["eprint_url"] == "https://arxiv.org/abs/2601.12345"
    assert "pub_url" not in records[1]
    json.dumps(payload, allow_nan=False)
    without_links = build_json.build_payload(frame.drop(columns=["pub_url", "eprint_url"]))
    assert len(without_links["records"]) == 2
