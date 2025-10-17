"""Build and persist publication embeddings, clusters, and labels.

This script builds a JSON payload describing the publication landscape. It:

1. Loads the dataset from Hugging Face.
2. Projects publication embeddings to two dimensions using t-SNE followed by PCA.
3. Clusters the projected points with DBSCAN, treating the ``-1`` label as noise.
4. Generates concise cluster labels with KeyBERT and stores the results as JSON.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import datasets
import numpy
import pandas
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

LOGGER = logging.getLogger(__name__)

DEFAULT_RANDOM_STATE = 42
KEYBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def compute_projection(
    embeddings: numpy.ndarray,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> numpy.ndarray:
    """Project embeddings to two dimensions using t-SNE followed by PCA.

    Args:
        embeddings: Matrix of shape ``(n_samples, n_features)``.
        random_state: Seed used to make t-SNE and PCA deterministic.

    Returns:
        A ``(n_samples, 2)`` array containing the projected coordinates.
    """

    if embeddings.ndim != 2:
        msg = "Embeddings must be a 2D array."
        raise ValueError(msg)

    n_samples = embeddings.shape[0]
    if n_samples < 2:
        msg = "At least two samples are required to compute a projection."
        raise ValueError(msg)

    perplexity = min(30, max(1, n_samples - 1))
    LOGGER.debug("Using t-SNE perplexity %s for %s samples", perplexity, n_samples)

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    tsne_embeddings = tsne.fit_transform(embeddings)

    pca = PCA(n_components=2, random_state=random_state)
    oriented_embeddings = pca.fit_transform(tsne_embeddings)
    return oriented_embeddings


def cluster_points(
    coordinates: numpy.ndarray,
    eps: float = 3.0,
    min_samples: int = 4,
) -> numpy.ndarray:
    """Cluster 2D coordinates using DBSCAN.

    Args:
        coordinates: ``(n_samples, 2)`` array of x/y pairs.
        eps: DBSCAN ``eps`` parameter controlling cluster diameter.
        min_samples: Minimum number of samples required to form a cluster.

    Returns:
        Array of cluster labels where ``-1`` denotes noise points.
    """

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        msg = "Coordinates must be a 2D array with two columns."
        raise ValueError(msg)

    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clusterer.fit_predict(coordinates)
    return labels


def ensure_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Ensure a sentence-transformer model is available locally.

    Args:
        model_name: Hugging Face model identifier passed to :class:`SentenceTransformer`.

    Returns:
        An instantiated :class:`SentenceTransformer` model.

    Raises:
        RuntimeError: If the model cannot be loaded, likely due to missing assets.
    """

    try:
        return SentenceTransformer(model_name)
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive
        msg = (
            "Failed to load sentence-transformer model '%s'. Ensure the model is "
            "downloaded and accessible in CI environments."
        )
        raise RuntimeError(msg % model_name) from exc


def _bib_dict_to_fragments(bib_entry: Dict[str, object]) -> List[str]:
    """Extract text fragments used to label clusters."""

    fragments: List[str] = []
    for key in ("title", "conference", "journal"):
        value = bib_entry.get(key) if isinstance(bib_entry, dict) else None
        if isinstance(value, str):
            fragments.append(value)
    return fragments


def build_cluster_text(records: pandas.DataFrame) -> str:
    """Aggregate descriptive text for a cluster.

    Args:
        records: Subset of the citations DataFrame belonging to a single cluster.

    Returns:
        Concatenated textual description for KeyBERT.
    """

    fragments: List[str] = []
    for bib_entry in records["bib_dict"]:
        fragments.extend(_bib_dict_to_fragments(bib_entry))

    unique_fragments = list(dict.fromkeys(fragments))
    return ". ".join(unique_fragments)


def extract_cluster_label(text: str, model: KeyBERT) -> str:
    """Extract a concise label for the supplied text using KeyBERT."""

    if not text:
        return ""

    keywords = model.extract_keywords(text, stop_words="english", top_n=1)
    if not keywords:
        return ""
    return keywords[0][0]


@dataclass
class ClusterSummary:
    """Structured summary of a cluster."""

    cluster_id: int
    label: str
    centroid_x: float
    centroid_y: float

    def as_dict(self) -> Dict[str, object]:
        """Serialize the cluster summary to a JSON-compatible dictionary."""

        return {
            "id": self.cluster_id,
            "label": self.label,
            "centroid": {"x": self.centroid_x, "y": self.centroid_y},
        }


def _cluster_id_series(labels: Sequence[int]) -> pandas.Series:
    """Convert raw DBSCAN labels into a pandas Series preserving ``None`` for noise."""

    return pandas.Series(
        [label if label != -1 else None for label in labels], dtype="object"
    )


def summarize_clusters(
    citations: pandas.DataFrame, labels: Sequence[int], label_model: KeyBERT
) -> List[ClusterSummary]:
    """Build cluster summaries including centroid coordinates and labels."""

    summaries: List[ClusterSummary] = []
    citations = citations.copy()
    citations["cluster_id"] = _cluster_id_series(labels)

    cluster_ids = sorted({label for label in labels if label != -1})
    LOGGER.debug("Building summaries for clusters: %s", cluster_ids)

    for cluster_id in cluster_ids:
        cluster_records = citations[citations["cluster_id"] == cluster_id]
        if cluster_records.empty:
            continue

        centroid_x = float(cluster_records["x"].mean())
        centroid_y = float(cluster_records["y"].mean())
        text = build_cluster_text(cluster_records)
        label = extract_cluster_label(text, label_model)
        summaries.append(
            ClusterSummary(
                cluster_id=int(cluster_id),
                label=label,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
            )
        )

    return summaries


def build_payload(citations: pandas.DataFrame, label_model: KeyBERT) -> Dict[str, object]:
    """Construct the JSON payload including records and cluster summaries."""

    embeddings = numpy.stack(citations["embedding"].values)
    coordinates = compute_projection(embeddings)
    labels = cluster_points(coordinates)
    citations = citations.copy()
    citations["x"] = coordinates[:, 0]
    citations["y"] = coordinates[:, 1]
    citations["cluster_id"] = _cluster_id_series(labels)

    summaries = summarize_clusters(citations, labels, label_model)

    records = citations[
        [
            "x",
            "y",
            "author_pub_id",
            "pub_year",
            "num_citations",
            "bib_dict",
            "cluster_id",
        ]
    ].to_dict(orient="records")

    payload = {
        "records": records,
        "clusters": [summary.as_dict() for summary in summaries],
    }
    return payload


def load_citations() -> pandas.DataFrame:
    """Load the publications dataset from Hugging Face."""

    citations = datasets.load_dataset("ccm/publications")["train"].to_pandas()
    citations["pub_year"] = [
        int(pub.get("pub_year", 2025)) if isinstance(pub, dict) else 2025
        for pub in citations["bib_dict"]
    ]
    return citations


def dump_payload(payload: Dict[str, object]) -> None:
    """Persist the payload to ``assets/json/pubs.json`` relative to the repo root."""

    workspace = os.environ.get("GITHUB_WORKSPACE", "..")
    output_dir = os.path.join(workspace, "assets/json")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pubs.json")

    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def main() -> None:
    """Entry point for building the publication JSON payload."""

    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Loading citations dataset")
    citations = load_citations()

    LOGGER.info("Loading KeyBERT model: %s", KEYBERT_MODEL_NAME)
    sentence_transformer = ensure_sentence_transformer(KEYBERT_MODEL_NAME)
    label_model = KeyBERT(model=sentence_transformer)

    LOGGER.info("Building JSON payload")
    payload = build_payload(citations, label_model)

    LOGGER.info("Persisting payload")
    dump_payload(payload)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
