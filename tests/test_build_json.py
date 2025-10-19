"""Tests for the publication clustering helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy
import pandas
import pytest
from keybert import KeyBERT

sys.path.append(str(Path(__file__).resolve().parents[1]))

from _scripts import build_json


@pytest.fixture(scope="session")
def fixture_records() -> list[dict[str, object]]:
    """Load the synthetic publication dataset for clustering tests."""

    fixture_path = Path(__file__).parent / "fixtures" / "cluster_fixture.json"
    return json.loads(fixture_path.read_text())


@pytest.fixture(scope="session")
def keybert_model() -> KeyBERT:
    """Instantiate a deterministic KeyBERT model for labelling tests."""

    transformer = build_json.ensure_sentence_transformer(build_json.KEYBERT_MODEL_NAME)
    return KeyBERT(model=transformer)


def test_cluster_points_assigns_clusters_and_noise(fixture_records: list[dict[str, object]]) -> None:
    """HDBSCAN should form clusters while marking distant points as noise."""

    coordinates = numpy.array([[record["x"], record["y"]] for record in fixture_records], dtype=float)
    labels = build_json.cluster_points(coordinates)

    assert labels[0] == labels[1] == labels[2] == labels[3]
    assert labels[4] == labels[5] == labels[6] == labels[7]
    assert labels[0] != labels[4]
    assert labels[-1] == -1


def test_cluster_points_limits_cluster_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DBSCAN clusterer should receive the tuned hyper-parameters."""

    coordinates = numpy.zeros((40, 2), dtype=float)
    captured_kwargs: dict[str, object] = {}

    class DummyClusterer:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

        def fit_predict(self, coords: numpy.ndarray) -> numpy.ndarray:
            return numpy.zeros(coords.shape[0], dtype=int)

    def fake_dbscan(*args: object, **kwargs: object) -> DummyClusterer:
        return DummyClusterer(**kwargs)

    monkeypatch.setattr(build_json, "DBSCAN", fake_dbscan)

    labels = build_json.cluster_points(coordinates)

    assert numpy.array_equal(labels, numpy.zeros(40, dtype=int))
    assert captured_kwargs["eps"] == pytest.approx(2.5)
    assert captured_kwargs["min_samples"] == 4


def test_noise_points_remain_unlabelled(fixture_records: list[dict[str, object]]) -> None:
    """Noise points (label ``-1``) should not retain a cluster identifier."""

    coordinates = numpy.array([[record["x"], record["y"]] for record in fixture_records], dtype=float)
    labels = build_json.cluster_points(coordinates)
    citations = pandas.DataFrame(fixture_records)
    citations["cluster_id"] = build_json._cluster_id_series(labels)

    noise_cluster_id = citations.loc[citations["author_pub_id"] == "pub-9", "cluster_id"].iloc[0]
    assert noise_cluster_id is None


def test_keybert_generates_deterministic_label(
    fixture_records: list[dict[str, object]], keybert_model: KeyBERT
) -> None:
    """KeyBERT should consistently return the same label for the fixture cluster."""

    robotics_cluster = pandas.DataFrame(
        [record for record in fixture_records if record["author_pub_id"] in {"pub-5", "pub-6", "pub-7", "pub-8"}]
    )
    text = build_json.build_cluster_text(robotics_cluster)

    first_label = build_json.extract_cluster_label(text, keybert_model)
    second_label = build_json.extract_cluster_label(text, keybert_model)

    assert first_label == "robot"
    assert first_label == second_label


def test_extract_pub_year_handles_missing_values() -> None:
    """Publication year extraction should default when the source is missing."""

    assert build_json._extract_pub_year({"pub_year": None}) == 2025
    assert build_json._extract_pub_year({}, default_year=1999) == 1999
    assert build_json._extract_pub_year("not-a-dict") == 2025
    assert build_json._extract_pub_year({"pub_year": " 2017 "}) == 2017
