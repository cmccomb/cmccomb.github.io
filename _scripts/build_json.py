"""Build and persist publication embeddings, clusters, and labels.

This module loads the publications dataset, projects the embeddings for
visualisation, clusters the items in a stable high-dimensional space, and
produces descriptive labels per cluster. The resulting payload is persisted to
``assets/json/pubs.json`` in an atomic manner so downstream consumers never
observe partial writes.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import logging
import os
import platform
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import datasets  # type: ignore[import-untyped]
import numpy
import pandas  # type: ignore[import-untyped]
from huggingface_hub import HfApi
from numpy.typing import NDArray
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.manifold import trustworthiness, TSNE  # type: ignore[import-untyped]

if __package__:
    from .derive_topic_labels import derive_labels, labeling_metadata, assign_unique_labels as _assign_unique_topic_labels
else:  # pragma: no cover - direct script execution
    from derive_topic_labels import derive_labels, labeling_metadata, assign_unique_labels as _assign_unique_topic_labels

LOGGER = logging.getLogger(__name__)

DEFAULT_RANDOM_STATE = 42
DEFAULT_DATASET_ID = "ccm/publications"
DEFAULT_DATASET_REVISION = "main"
DEFAULT_KMEANS_CLUSTERS = 10
DEFAULT_KMEANS_N_INIT = 100
DEFAULT_PERPLEXITY = 40
DEFAULT_TSNE_ITERATIONS = 2000


@dataclass(frozen=True)
class ProjectionResult:
    """Container for projection coordinates and provenance."""

    coordinates: numpy.ndarray
    perplexity: int
    preprocessing: str = "l2+pca50(full)"
    explained_variance: float | None = None
    kl_divergence: float | None = None


@dataclass(frozen=True)
class ReducedSpace:
    """Description of the space used for clustering."""

    matrix: numpy.ndarray
    descriptor: str


@dataclass(frozen=True)
class ClusteringResult:
    """Result of clustering the embeddings."""

    labels: numpy.ndarray
    space: str
    algorithm: str
    n_clusters: int


def _datasets_offline() -> bool:
    """Return whether Hugging Face datasets should avoid network access."""

    value = os.environ.get("HF_DATASETS_OFFLINE", "0").lower()
    return value in {"1", "true", "yes"}

def compute_projection(
    embeddings: numpy.ndarray,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ProjectionResult:
    """Visualize cosine neighborhoods with PCA-preprocessed, PCA-initialized t-SNE."""

    if embeddings.ndim != 2 or embeddings.shape[1] < 2:
        raise ValueError("Embeddings must be a 2D array with at least two features.")
    embeddings = numpy.asarray(embeddings, dtype=numpy.float64)
    n_samples, n_features = embeddings.shape
    if n_samples < 3:
        raise ValueError("At least three samples are required for t-SNE.")
    if not numpy.isfinite(embeddings).all() or numpy.any(numpy.linalg.norm(embeddings, axis=1) == 0):
        raise ValueError("Embeddings must be finite and nonzero.")

    unit = normalize(embeddings)
    components = min(50, n_samples - 1, n_features)
    denoising = PCA(n_components=components, svd_solver="full")
    reduced = denoising.fit(unit).transform(unit)
    # A full-SVD start avoids randomized PCA initialization changing the map
    # when the seed changes. Match t-SNE's small initial coordinate scale.
    initial = PCA(n_components=2, svd_solver="full").fit_transform(unit).astype(numpy.float32)
    scale = float(numpy.std(initial[:, 0]))
    if scale > 0:
        initial = initial / numpy.std(initial[:, 0]) * 1e-4
    perplexity = min(DEFAULT_PERPLEXITY, max(2, (n_samples - 1) // 3))
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        init=initial,
        metric="cosine",
        learning_rate="auto",
        max_iter=DEFAULT_TSNE_ITERATIONS,
        n_jobs=1,
    )
    coordinates = PCA(n_components=2, svd_solver="full").fit_transform(tsne.fit_transform(reduced))
    return ProjectionResult(
        coordinates=coordinates,
        perplexity=perplexity,
        preprocessing=f"l2+pca{components}(full)",
        explained_variance=float(denoising.explained_variance_ratio_.sum()),
        kl_divergence=float(tsne.kl_divergence_),
    )


def reduce_for_clustering(
    embeddings: numpy.ndarray, random_state: int = DEFAULT_RANDOM_STATE
) -> ReducedSpace:
    """Denoise semantic embeddings without using visualization coordinates."""

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    embeddings = numpy.asarray(embeddings, dtype=numpy.float64)
    n_samples, n_features = embeddings.shape
    if n_features <= 50:
        return ReducedSpace(matrix=embeddings, descriptor="original")
    components = min(50, n_samples - 1, n_features)
    reduced = PCA(n_components=components, svd_solver="full").fit_transform(embeddings)
    return ReducedSpace(matrix=reduced, descriptor=f"pca{components}")


def cluster_points_from_embeddings(
    embeddings: numpy.ndarray,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ClusteringResult:
    """Group papers in semantic space, independently of the t-SNE display."""

    reduced = reduce_for_clustering(embeddings, random_state)
    clusterer = KMeans(
        n_clusters=min(DEFAULT_KMEANS_CLUSTERS, len(embeddings)),
        n_init=DEFAULT_KMEANS_N_INIT,
        random_state=random_state,
    )
    labels = clusterer.fit_predict(reduced.matrix)
    return ClusteringResult(
        labels=numpy.asarray(labels, dtype=int),
        space=reduced.descriptor,
        algorithm="kmeans",
        n_clusters=int(clusterer.n_clusters),
    )


def map_quality(embeddings: numpy.ndarray, coordinates: numpy.ndarray, labels: numpy.ndarray) -> dict:
    """Measure semantic grouping and local map fidelity in original cosine space."""

    embeddings = numpy.asarray(embeddings, dtype=numpy.float64)
    samples = len(embeddings)
    neighbors = min(10, samples - 1)
    distances = pairwise_distances(embeddings, metric="cosine")
    numpy.fill_diagonal(distances, numpy.inf)
    original = distances.argsort(axis=1)[:, :neighbors]
    shown = pairwise_distances(coordinates)
    numpy.fill_diagonal(shown, numpy.inf)
    projected = shown.argsort(axis=1)[:, :neighbors]
    trust_neighbors = min(10, (samples - 1) // 2)
    groups = len(numpy.unique(labels))
    return {
        "reference_metric": "cosine(original embeddings)",
        "neighbors": neighbors,
        "cosine_silhouette": float(silhouette_score(embeddings, labels, metric="cosine"))
            if 1 < groups < samples else None,
        "semantic_neighbor_agreement": float(numpy.mean(labels[original] == labels[:, None])),
        "projection_neighbor_recall": float(numpy.mean([
            len(set(a) & set(b)) / neighbors for a, b in zip(original, projected)
        ])) if neighbors else None,
        "trustworthiness_neighbors": trust_neighbors,
        "projection_trustworthiness": float(trustworthiness(
            embeddings, coordinates, n_neighbors=trust_neighbors, metric="cosine"
        )) if trust_neighbors else None,
    }


def _bib_dict_to_fragments(bib_entry: Dict[str, object]) -> List[str]:
    """Extract text fragments used to label clusters."""

    fragments: List[str] = []
    for key in ("title", "abstract"):
        value = bib_entry.get(key) if isinstance(bib_entry, dict) else None
        if isinstance(value, str) and value:
            fragments.append(value)
    return fragments


def build_cluster_text(records: pandas.DataFrame, max_chars: int = 100_000) -> str:
    """Aggregate descriptive text for a cluster."""

    fragments: List[str] = []
    seen: set[str] = set()
    total_chars = 0
    for bib_entry in records["bib_dict"]:
        for fragment in _bib_dict_to_fragments(bib_entry):
            if fragment in seen:
                continue
            seen.add(fragment)
            fragments.append(fragment)
            total_chars += len(fragment)
            if total_chars >= max_chars:
                break
        if total_chars >= max_chars:
            break
    return ". ".join(fragments)


def _cluster_id_series(labels: Sequence[int]) -> pandas.Series:
    """Convert raw clustering labels into a Series preserving ``None`` for noise."""

    label_list = [int(label) for label in labels]
    return pandas.Series(
        [label if label != -1 else None for label in label_list], dtype="object"
    )


def ctfidf_labels(citations: pandas.DataFrame, labels: Sequence[int]) -> Dict[int, str]:
    """Derive complete topic phrases using paper-level title/abstract evidence."""

    return derive_labels(citations.to_dict(orient="records"), [int(label) for label in labels])


def relabel_payload(payload: Dict[str, object]) -> Dict[str, object]:
    """Regenerate titles from an existing snapshot, preserving its paper layout."""

    updated = deepcopy(payload)
    records = updated["records"]
    labels = [int(record["cluster_id"]) if record.get("cluster_id") is not None else -1
              for record in records]
    titles = derive_labels(records, labels)
    for cluster in updated["clusters"]:
        cluster["label"] = titles[cluster["id"]]
    updated["meta"]["labeling"] = labeling_metadata()
    return updated


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


def summarize_clusters(
    citations: pandas.DataFrame,
    labels: Sequence[int],
    ctfidf: Dict[int, str] | None = None,
) -> List[ClusterSummary]:
    """Build cluster summaries including centroid coordinates and labels."""

    label_list = [int(label) for label in labels]
    summaries: List[ClusterSummary] = []
    citations = citations.copy()
    citations["cluster_id"] = _cluster_id_series(label_list)
    cluster_ids = sorted({label for label in label_list if label != -1})
    LOGGER.debug("Building summaries for clusters: %s", cluster_ids)

    ctfidf = ctfidf or {}

    for cluster_id in cluster_ids:
        cluster_records = citations[citations["cluster_id"] == cluster_id]
        if cluster_records.empty:
            continue

        centroid_x = float(cluster_records["x"].mean())
        centroid_y = float(cluster_records["y"].mean())
        label = ctfidf.get(cluster_id, "")

        summaries.append(
            ClusterSummary(
                cluster_id=int(cluster_id),
                label=label,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
            )
        )

    return summaries


def _extract_pub_year(bib_entry: object, default_year: int | None = None) -> int:
    """Return the publication year encoded in a bibliographic entry."""

    if default_year is None:
        default_year = datetime.now(timezone.utc).year

    if not isinstance(bib_entry, dict):
        return default_year

    raw_year = bib_entry.get("pub_year")
    if raw_year is None:
        return default_year

    if isinstance(raw_year, (int, float)):
        return int(raw_year)

    if isinstance(raw_year, str):
        stripped_year = raw_year.strip()
        if stripped_year:
            try:
                return int(stripped_year)
            except ValueError:  # pragma: no cover - defensive guard
                return default_year

    return default_year


def load_citations(
    dataset_id: str = DEFAULT_DATASET_ID,
    revision: str | None = DEFAULT_DATASET_REVISION,
) -> pandas.DataFrame:
    """Load the publications dataset from Hugging Face."""

    load_kwargs: Dict[str, object] = {"revision": revision} if revision else {}
    if _datasets_offline():
        load_kwargs["download_mode"] = datasets.DownloadMode.REUSE_DATASET_IF_EXISTS

    try:
        dataset = datasets.load_dataset(dataset_id, **load_kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"Failed to load dataset '{dataset_id}' (revision={revision}): {exc}"
        raise RuntimeError(msg) from exc

    table = dataset["train"].to_pandas()
    required_columns = {"embedding", "bib_dict", "author_pub_id", "num_citations"}
    missing = required_columns - set(table.columns)
    if missing:
        msg = f"Dataset missing required columns: {sorted(missing)}"
        raise RuntimeError(msg)

    try:
        _ = numpy.stack(table["embedding"].values)
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"Embeddings column not uniform arrays: {exc}"
        raise RuntimeError(msg) from exc

    table["pub_year"] = [
        _extract_pub_year(bib_entry) for bib_entry in table["bib_dict"]
    ]
    return table


def resolve_dataset_commit(
    dataset_id: str = DEFAULT_DATASET_ID,
    revision: str | None = DEFAULT_DATASET_REVISION,
) -> str | None:
    """Resolve a mutable Hugging Face revision to its immutable commit SHA."""

    if not revision or _datasets_offline():
        return revision

    try:
        return HfApi().dataset_info(dataset_id, revision=revision).sha
    except Exception as exc:  # pragma: no cover - network-dependent fallback
        LOGGER.warning("Could not resolve dataset commit: %s", exc)
        return revision


def _hash_bytes(payload: bytes) -> str:
    """Return the SHA-256 hash for the provided payload."""

    return hashlib.sha256(payload).hexdigest()


def dump_payload(payload: Dict[str, object], force: bool = False) -> bool:
    """Persist the payload to ``assets/json/pubs.json`` using an atomic write."""

    workspace = Path(
        os.environ.get("GITHUB_WORKSPACE", Path(__file__).resolve().parents[1])
    )
    output_dir = workspace / "assets/json"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "pubs.json"

    payload_bytes = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload_bytes_with_newline = payload_bytes + b"\n"

    if not force and os.path.exists(output_path):
        with open(output_path, "rb") as file_handle:
            existing_bytes = file_handle.read()
        if _hash_bytes(existing_bytes) == _hash_bytes(payload_bytes_with_newline):
            LOGGER.info("No changes in payload; skipping write.")
            return False

    fd, temp_path = tempfile.mkstemp(dir=output_dir, prefix="pubs.", suffix=".tmp")
    with os.fdopen(fd, "wb") as temp_file:
        temp_file.write(payload_bytes_with_newline)
    os.replace(temp_path, output_path)
    LOGGER.info("Wrote payload to %s", output_path)
    return True


def curation_metadata(
    *,
    random_state: int,
    projection: ProjectionResult,
    clustering: ClusteringResult,
    dataset_id: str,
    dataset_revision: str | None,
    dataset_commit: str | None,
    noise_fraction: float,
    record_count: int,
) -> Dict[str, object]:
    """Build a metadata dictionary describing the build configuration."""

    import sklearn  # type: ignore[import-untyped]

    return {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "labeling": labeling_metadata(),
        "random_state": random_state,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "dataset_commit": dataset_commit,
        "projection": {
            "method": "tsne+pca",
            "perplexity": int(projection.perplexity),
            "preprocessing": projection.preprocessing,
            "pca_explained_variance": projection.explained_variance,
            "metric": "cosine",
            "initialization": "pca(full), scaled to 1e-4",
            "max_iter": DEFAULT_TSNE_ITERATIONS,
            "learning_rate": "auto",
            "kl_divergence": projection.kl_divergence,
        },
        "clustering": {
            "space": clustering.space,
            "algorithm": clustering.algorithm,
            "n_clusters": int(clustering.n_clusters),
            "n_init": DEFAULT_KMEANS_N_INIT,
            "projection_refinement": False,
        },
        "noise_fraction": noise_fraction,
        "record_count": record_count,
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "sklearn": sklearn.__version__,
            "datasets": datasets.__version__,
        },
    }


def build_payload(
    citations: pandas.DataFrame,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_revision: str | None = DEFAULT_DATASET_REVISION,
    dataset_commit: str | None = None,
) -> Dict[str, object]:
    """Construct the JSON payload including records and cluster summaries."""

    embeddings = numpy.stack(citations["embedding"].values)
    projection = compute_projection(embeddings, random_state=random_state)
    clustering = cluster_points_from_embeddings(
        embeddings,
        random_state=random_state,
    )

    coordinates = projection.coordinates
    labels_array: NDArray[numpy.int_] = numpy.asarray(clustering.labels, dtype=int)
    label_sequence = [int(label) for label in labels_array.tolist()]
    citations = citations.copy()
    citations["x"] = coordinates[:, 0]
    citations["y"] = coordinates[:, 1]
    citations["cluster_id"] = _cluster_id_series(label_sequence)

    ctfidf = ctfidf_labels(citations, label_sequence)
    summaries = summarize_clusters(citations, label_sequence, ctfidf=ctfidf)

    record_columns = [
        "x",
        "y",
        "author_pub_id",
        "pub_year",
        "num_citations",
        "bib_dict",
        "cluster_id",
    ]
    # Preserve source links so each quarterly refresh retains direct paper access.
    resource_columns = [name for name in ("pub_url", "eprint_url", "doi") if name in citations]
    records = citations[record_columns + resource_columns].to_dict(orient="records")
    for record in records:
        for name in resource_columns:
            if not isinstance(record.get(name), str) or not record[name].strip():
                record.pop(name, None)

    noise_fraction = float(numpy.mean(labels_array == -1)) if len(labels_array) else 0.0

    payload = {
        "meta": curation_metadata(
            random_state=random_state,
            projection=projection,
            clustering=clustering,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            dataset_commit=dataset_commit,
            noise_fraction=noise_fraction,
            record_count=len(records),
        ),
        "records": records,
        "clusters": [summary.as_dict() for summary in summaries],
    }
    payload["meta"]["quality"] = map_quality(embeddings, coordinates, labels_array)
    return payload


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the build script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID)
    parser.add_argument("--revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--relabel-only", action="store_true",
                        help="Regenerate titles from the committed snapshot without fetching or reclustering")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Entry point for building the publication JSON payload."""

    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.relabel_only:
        workspace = Path(os.environ.get("GITHUB_WORKSPACE", Path(__file__).resolve().parents[1]))
        path = workspace / "assets/json/pubs.json"
        payload = relabel_payload(json.loads(path.read_text(encoding="utf-8")))
        if not args.dry_run:
            dump_payload(payload, force=args.force)
        return 0

    random.seed(args.seed)
    numpy.random.seed(args.seed)

    LOGGER.info(
        "Loading citations dataset %s (revision=%s)", args.dataset, args.revision
    )
    citations = load_citations(dataset_id=args.dataset, revision=args.revision)
    dataset_commit = resolve_dataset_commit(args.dataset, args.revision)

    LOGGER.info("Building JSON payload")
    payload = build_payload(
        citations,
        random_state=args.seed,
        dataset_id=args.dataset,
        dataset_revision=args.revision,
        dataset_commit=dataset_commit,
    )

    if args.dry_run:
        LOGGER.info("Dry run requested; skipping write")
        return 0

    LOGGER.info("Persisting payload")
    wrote = dump_payload(payload, force=args.force)
    return 0 if wrote or args.force else 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
