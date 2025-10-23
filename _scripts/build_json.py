"""Build and persist publication embeddings, clusters, and labels.

This module loads the publications dataset, projects the embeddings for
visualisation, clusters the items in a stable high-dimensional space, and
produces descriptive labels per cluster. The resulting payload is persisted to
``assets/json/pubs.json`` in an atomic manner so downstream consumers never
observe partial writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Protocol, Sequence, TYPE_CHECKING

import datasets  # type: ignore[import-untyped]
import numpy
import pandas  # type: ignore[import-untyped]
from huggingface_hub import snapshot_download
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN  # type: ignore[import-untyped]
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # type: ignore[import-untyped]
from sklearn.manifold import TSNE  # type: ignore[import-untyped]
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]

try:  # pragma: no cover - optional dependency
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - defensive guard
    hdbscan = None


if TYPE_CHECKING:  # pragma: no cover - imported for static type checking only
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]


class KeywordModel(Protocol):
    """Protocol describing the keyword extraction interface."""

    def extract_keywords(self, text: str, **kwargs: Any) -> List[tuple[str, float]]:
        """Return ranked keywords for the supplied text."""


LOGGER = logging.getLogger(__name__)

DEFAULT_RANDOM_STATE = 42
KEYBERT_MODEL_NAME = "allenai/specter2"
SPECTER2_BASE_MODEL_NAME = "allenai/specter2_base"
DEFAULT_DATASET_ID = "ccm/publications"
DEFAULT_DATASET_REVISION = "main"
AUTOTUNE_DISTANCE_QUANTILE = 0.90
EPSILON_ADJUSTMENT_FACTOR = 0.18

@dataclass(frozen=True)
class ProjectionResult:
    """Container for projection coordinates and provenance."""

    coordinates: numpy.ndarray
    perplexity: int


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
    eps: float | None
    min_samples: int


def _transformers_offline() -> bool:
    """Return whether transformers assets must be loaded from a local cache."""

    value = os.environ.get("TRANSFORMERS_OFFLINE", "0").lower()
    return value in {"1", "true", "yes"}


def _datasets_offline() -> bool:
    """Return whether Hugging Face datasets should avoid network access."""

    value = os.environ.get("HF_DATASETS_OFFLINE", "0").lower()
    return value in {"1", "true", "yes"}


def _build_specter2_sentence_transformer() -> "SentenceTransformer":
    """Construct a SentenceTransformer instance with the SPECTER2 adapter."""

    from adapters import AutoAdapterModel  # type: ignore[import-untyped]
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    from sentence_transformers import models as st_models  # type: ignore[import-untyped]

    offline = _transformers_offline()

    try:
        base_model_path = snapshot_download(
            SPECTER2_BASE_MODEL_NAME,
            local_files_only=offline,
        )
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive
        msg = (
            "Failed to load the SPECTER2 base model '%s'. Ensure the weights are "
            "available locally before running in offline environments."
        )
        raise RuntimeError(msg % SPECTER2_BASE_MODEL_NAME) from exc

    try:
        transformer = st_models.Transformer(base_model_path)
        adapter_model = AutoAdapterModel.from_pretrained(base_model_path)
        adapter_path = snapshot_download(
            KEYBERT_MODEL_NAME,
            local_files_only=offline,
        )
        adapter_name = adapter_model.load_adapter(
            adapter_path,
            source="local",
            load_as=KEYBERT_MODEL_NAME,
        )
        adapter_model.set_active_adapters(adapter_name)
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive
        msg = (
            "Failed to load the SPECTER2 adapter '%s'. Download the adapter "
            "weights and retry the build."
        )
        raise RuntimeError(msg % KEYBERT_MODEL_NAME) from exc

    transformer.auto_model = adapter_model

    pooling = st_models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
    )
    normalize = st_models.Normalize()
    return SentenceTransformer(modules=[transformer, pooling, normalize])


def ensure_sentence_transformer(model_name: str) -> "SentenceTransformer":
    """Ensure a sentence-transformer model is available locally."""

    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    try:
        if model_name == KEYBERT_MODEL_NAME:
            return _build_specter2_sentence_transformer()

        return SentenceTransformer(
            model_name,
            local_files_only=_transformers_offline(),
        )
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive
        msg = (
            "Failed to load sentence-transformer model '%s'. Ensure the model is "
            "downloaded and accessible in CI environments."
        )
        raise RuntimeError(msg % model_name) from exc


def compute_projection(
    embeddings: numpy.ndarray,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ProjectionResult:
    """Project embeddings to two dimensions using t-SNE followed by PCA."""

    if embeddings.ndim != 2:
        msg = "Embeddings must be a 2D array."
        raise ValueError(msg)

    n_samples = embeddings.shape[0]
    if n_samples < 3:
        msg = "At least three samples are required for t-SNE."
        raise ValueError(msg)

    max_ok = max(2, (n_samples - 1) // 3)
    perplexity = int(min(30, max(2, max_ok)))
    LOGGER.debug("Using t-SNE perplexity %d for %d samples", perplexity, n_samples)

    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
    )
    tsne_embeddings = tsne.fit_transform(embeddings)

    pca = PCA(n_components=2, random_state=random_state)
    oriented_embeddings = pca.fit_transform(tsne_embeddings)
    return ProjectionResult(coordinates=oriented_embeddings, perplexity=perplexity)


def reduce_for_clustering(
    embeddings: numpy.ndarray, random_state: int = DEFAULT_RANDOM_STATE
) -> ReducedSpace:
    """Reduce embeddings for clustering with PCA."""

    if embeddings.ndim != 2:
        msg = "Embeddings must be a 2D array."
        raise ValueError(msg)

    n_samples, n_features = embeddings.shape
    if n_features <= 50:
        return ReducedSpace(matrix=embeddings, descriptor="original")

    n_components = min(50, n_samples, n_features)
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)
    descriptor = f"pca{n_components}"
    return ReducedSpace(matrix=reduced, descriptor=descriptor)


def autotune_dbscan_eps(
    data: numpy.ndarray,
    k: int = 4,
    quantile: float = 0.95,
) -> float:
    """Estimate an ``eps`` value for DBSCAN via the k-NN distance heuristic."""

    if data.ndim != 2:
        msg = "Data must be a 2D array to estimate DBSCAN eps."
        raise ValueError(msg)

    n_samples = data.shape[0]
    if n_samples == 0:
        msg = "At least one sample is required to estimate eps."
        raise ValueError(msg)

    effective_k = min(max(1, k), n_samples)
    nbrs = NearestNeighbors(n_neighbors=effective_k)
    dists, _ = nbrs.fit(data).kneighbors(data)
    kth = numpy.sort(dists[:, -1])
    eps = float(numpy.quantile(kth, quantile))
    if eps == 0.0:
        eps = float(numpy.finfo(data.dtype).eps)
    return eps


def cluster_points_from_embeddings(
    embeddings: numpy.ndarray, random_state: int = DEFAULT_RANDOM_STATE
) -> ClusteringResult:
    """Cluster embeddings using DBSCAN in a reduced-dimensional space."""

    reduced = reduce_for_clustering(embeddings, random_state)
    min_samples = min(4, embeddings.shape[0])
    min_samples = max(1, min_samples)
    eps = autotune_dbscan_eps(
        reduced.matrix,
        k=min_samples,
        quantile=AUTOTUNE_DISTANCE_QUANTILE,
    ) * EPSILON_ADJUSTMENT_FACTOR
    LOGGER.info("DBSCAN eps=%.3f (auto), min_samples=%d", eps, min_samples)
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = clusterer.fit_predict(reduced.matrix)
    algorithm = "dbscan"
    eps_used: float | None = eps

    if numpy.all(labels == -1) and hdbscan is not None and reduced.matrix.shape[0] >= 2:
        LOGGER.info("DBSCAN produced only noise; retrying with HDBSCAN.")
        clusterer_hdb = hdbscan.HDBSCAN(min_cluster_size=max(min_samples, 5))
        labels = clusterer_hdb.fit_predict(reduced.matrix)
        algorithm = "hdbscan"
        eps_used = None

    return ClusteringResult(
        labels=labels,
        space=reduced.descriptor,
        algorithm=algorithm,
        eps=eps_used,
        min_samples=min_samples,
    )


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


def extract_cluster_label(text: str, model: KeywordModel) -> str:
    """Extract a concise label for the supplied text using KeyBERT."""

    if not text:
        return ""

    keywords = model.extract_keywords(
        text,
        stop_words="english",
        use_mmr=True,
        diversity=0.7,
        top_n=3,
        keyphrase_ngram_range=(2, 3),
    )
    phrases = [keyword for keyword, _ in keywords]
    return ", ".join(phrases)


def _cluster_id_series(labels: Sequence[int]) -> pandas.Series:
    """Convert raw clustering labels into a Series preserving ``None`` for noise."""

    label_list = [int(label) for label in labels]
    return pandas.Series(
        [label if label != -1 else None for label in label_list], dtype="object"
    )


def ctfidf_labels(
    citations: pandas.DataFrame, labels: Sequence[int], top_k: int = 3
) -> Dict[int, str]:
    """Generate cluster labels using class-based TF-IDF."""

    label_list = [int(label) for label in labels]
    df = citations.copy()
    df["cluster_id"] = _cluster_id_series(label_list)
    cluster_ids = sorted({label for label in label_list if label != -1})
    if not cluster_ids:
        return {}

    docs: List[str] = []
    for cluster_id in cluster_ids:
        cluster_records = df[df["cluster_id"] == cluster_id]
        titles: List[str] = []
        abstracts: List[str] = []
        for bib_entry in cluster_records["bib_dict"]:
            if isinstance(bib_entry, dict):
                title = bib_entry.get("title")
                abstract = bib_entry.get("abstract")
                if isinstance(title, str):
                    titles.append(title)
                if isinstance(abstract, str):
                    abstracts.append(abstract)
        text = ((". ".join(titles) + ". ") * 2) + ". ".join(abstracts)
        docs.append(text)

    if not docs:
        return {}

    vectorizer = CountVectorizer(
        ngram_range=(2, 3),
        stop_words="english",
        min_df=1,
        max_df=0.9,
    )

    try:
        term_matrix = vectorizer.fit_transform(docs)
    except ValueError:
        return {}

    tfidf = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)
    tfidf_matrix = tfidf.fit_transform(term_matrix)
    vocabulary = numpy.array(vectorizer.get_feature_names_out())

    labels_out: Dict[int, str] = {}
    for index, cluster_id in enumerate(cluster_ids):
        row = tfidf_matrix[index].toarray().ravel()
        if row.sum() == 0:
            continue
        top_indices = row.argsort()[-top_k:][::-1]
        labels_out[cluster_id] = ", ".join(vocabulary[top_indices])

    return labels_out


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
    label_model: KeywordModel,
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
        if not label:
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


def _extract_pub_year(bib_entry: object, default_year: int = 2025) -> int:
    """Return the publication year encoded in a bibliographic entry."""

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


def _hash_bytes(payload: bytes) -> str:
    """Return the SHA-256 hash for the provided payload."""

    return hashlib.sha256(payload).hexdigest()


def dump_payload(payload: Dict[str, object], force: bool = False) -> bool:
    """Persist the payload to ``assets/json/pubs.json`` using an atomic write."""

    workspace = os.environ.get("GITHUB_WORKSPACE", "..")
    output_dir = os.path.join(workspace, "assets/json")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pubs.json")

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
    noise_fraction: float,
    record_count: int,
) -> Dict[str, object]:
    """Build a metadata dictionary describing the build configuration."""

    import keybert  # type: ignore[import-untyped]
    import sentence_transformers  # type: ignore[import-untyped]
    import sklearn  # type: ignore[import-untyped]

    return {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "projection": {
            "method": "tsne+pca",
            "perplexity": int(projection.perplexity),
        },
        "clustering": {
            "space": clustering.space,
            "algorithm": clustering.algorithm,
            "eps": float(clustering.eps) if clustering.eps is not None else None,
            "min_samples": int(clustering.min_samples),
        },
        "noise_fraction": noise_fraction,
        "record_count": record_count,
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "sklearn": sklearn.__version__,
            "keybert": keybert.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "datasets": datasets.__version__,
        },
    }


def build_payload(
    citations: pandas.DataFrame,
    label_model: KeywordModel,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_revision: str | None = DEFAULT_DATASET_REVISION,
) -> Dict[str, object]:
    """Construct the JSON payload including records and cluster summaries."""

    embeddings = numpy.stack(citations["embedding"].values)
    clustering = cluster_points_from_embeddings(embeddings, random_state=random_state)
    projection = compute_projection(embeddings, random_state=random_state)

    coordinates = projection.coordinates
    labels_array: NDArray[numpy.int_] = numpy.asarray(clustering.labels, dtype=int)
    label_sequence = [int(label) for label in labels_array.tolist()]
    citations = citations.copy()
    citations["x"] = coordinates[:, 0]
    citations["y"] = coordinates[:, 1]
    citations["cluster_id"] = _cluster_id_series(label_sequence)

    ctfidf = ctfidf_labels(citations, label_sequence, top_k=2)
    summaries = summarize_clusters(
        citations, label_sequence, label_model, ctfidf=ctfidf
    )

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

    noise_fraction = float(numpy.mean(labels_array == -1)) if len(labels_array) else 0.0

    payload = {
        "meta": curation_metadata(
            random_state=random_state,
            projection=projection,
            clustering=clustering,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            noise_fraction=noise_fraction,
            record_count=len(records),
        ),
        "records": records,
        "clusters": [summary.as_dict() for summary in summaries],
    }
    return payload


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the build script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID)
    parser.add_argument("--revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Entry point for building the publication JSON payload."""

    from keybert import KeyBERT  # type: ignore[import-untyped]

    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    random.seed(args.seed)
    numpy.random.seed(args.seed)

    LOGGER.info(
        "Loading citations dataset %s (revision=%s)", args.dataset, args.revision
    )
    citations = load_citations(dataset_id=args.dataset, revision=args.revision)

    LOGGER.info("Loading KeyBERT model: %s", KEYBERT_MODEL_NAME)
    sentence_transformer = ensure_sentence_transformer(KEYBERT_MODEL_NAME)
    label_model = KeyBERT(model=sentence_transformer)

    LOGGER.info("Building JSON payload")
    payload = build_payload(
        citations,
        label_model,
        random_state=args.seed,
        dataset_id=args.dataset,
        dataset_revision=args.revision,
    )

    if args.dry_run:
        LOGGER.info("Dry run requested; skipping write")
        return 0

    LOGGER.info("Persisting payload")
    wrote = dump_payload(payload, force=args.force)
    return 0 if wrote or args.force else 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
