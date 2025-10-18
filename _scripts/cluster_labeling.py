"""Cluster labeling utilities using SPECTER2 embeddings.

This module implements an opinionated pipeline for extracting cluster labels
from collections of scientific documents. The pipeline follows the
requirements outlined in the Codex task description:

* SPECTER2 (``allenai/specter2_base``) embeddings computed in batches.
* Candidate generation via spaCy noun chunks constrained to the
  ``<J.*>*<N.*>+`` part-of-speech pattern.
* Candidate filtering with minimum document frequency, token limits, and
  removal of stop words and punctuation.
* Candidate deduplication using cosine similarity to collapse near-duplicate
  phrases.
* Cluster centroid computation with composite scoring that combines PMI / TF
  IDF (≈30%) and cosine similarity to the centroid (≈70%).
* Maximal marginal relevance (MMR) based diversification with an optional
  MaxSum re-ranking stage.
* Cohesion scoring for the resulting cluster label set.

The :func:`label_cluster` function is the main entry point and returns a JSON
serialisable dictionary of top phrases, the best phrase, and optional score
metadata.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Sequence

import numpy as np

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Span


LOGGER = logging.getLogger(__name__)


def _normalise_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalise vectors to unit length.

    Parameters
    ----------
    vectors:
        A two-dimensional array of embeddings.

    Returns
    -------
    numpy.ndarray
        The L2-normalised embeddings. If a vector has zero magnitude the
        original vector is returned unchanged.
    """

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        return vectors / safe_norms
class EmbeddingBackend(Protocol):
    """Protocol describing the interface required for embeddings."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return embeddings for the provided texts."""


class Specter2Embedder:
    """Embedding backend backed by the ``allenai/specter2_base`` checkpoint.

    The embedder lazily imports the ``transformers`` dependency to avoid
    imposing an import cost on clients that inject a custom backend for tests
    or other runtime environments.
    """

    def __init__(self, model_name: str = "allenai/specter2_base", batch_size: int = 8, device: Optional[str] = None) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._tokenizer = None
        self._model = None

    def _ensure_model(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        if self._tokenizer is None or self._model is None:
            LOGGER.info("Loading SPECTER2 model '%s'", self._model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name)
            if self._device is not None:
                self._model.to(self._device)
            self._model.eval()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return SPECTER2 embeddings for ``texts``.

        Parameters
        ----------
        texts:
            Sequence of raw document texts.

        Returns
        -------
        numpy.ndarray
            ``(len(texts), hidden_size)`` embedding matrix.
        """

        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        self._ensure_model()

        assert self._tokenizer is not None
        assert self._model is not None

        embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), self._batch_size):
            batch_texts = list(texts[start : start + self._batch_size])
            encoded = self._tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            if self._device is not None:
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with np.errstate(over="ignore"):
                outputs = self._model(**encoded)
            token_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings.append(token_embeddings)
        return np.vstack(embeddings)


@dataclass(frozen=True)
class Candidate:
    """Container describing a keyphrase candidate."""

    text: str
    term_frequency: int
    document_frequency: int


class KeyphraseCandidateGenerator(Protocol):
    """Protocol for generating keyphrase candidates from documents."""

    def generate(self, documents: Sequence[str]) -> List[Candidate]:
        """Return a list of unique :class:`Candidate` objects."""


class NounPhraseCandidateGenerator:
    """Generate keyphrase candidates via spaCy noun-chunk extraction.

    Parameters
    ----------
    nlp:
        Optional spaCy language model. When ``None`` a small English model is
        loaded on demand.
    min_document_frequency:
        Minimum number of documents in which a candidate must appear to be
        retained.
    min_ngram:
        Minimum token length (inclusive) for a candidate phrase.
    max_ngram:
        Maximum token length (inclusive) for a candidate phrase.
    """

    def __init__(
        self,
        nlp: Optional["Language"] = None,
        *,
        min_document_frequency: int = 2,
        min_ngram: int = 2,
        max_ngram: int = 4,
    ) -> None:
        self._provided_nlp = nlp
        self._min_document_frequency = min_document_frequency
        self._min_ngram = min_ngram
        self._max_ngram = max_ngram

    def _ensure_nlp(self) -> "Language":
        import spacy

        if self._provided_nlp is None:
            LOGGER.info("Loading spaCy English model for noun-chunk extraction")
            self._provided_nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        return self._provided_nlp

    @staticmethod
    def _is_valid_chunk(chunk: "Span") -> bool:
        tokens = [token for token in chunk if not (token.is_stop or token.is_punct or token.is_space)]
        if len(tokens) == 0:
            return False
        pos_pattern = [token.tag_ for token in tokens]
        has_valid_pattern = all(tag.startswith("J") or tag.startswith("N") for tag in pos_pattern)
        has_terminal_noun = any(tag.startswith("N") for tag in pos_pattern[-1:])
        if not (has_valid_pattern and has_terminal_noun):
            return False
        return True

    def generate(self, documents: Sequence[str]) -> List[Candidate]:
        from collections import Counter, defaultdict

        if not documents:
            return []

        nlp = self._ensure_nlp()
        doc_counts: Dict[str, Counter[int]] = defaultdict(Counter)
        term_counts: Dict[str, int] = defaultdict(int)

        for doc_index, text in enumerate(documents):
            spacy_doc = nlp(text)
            seen_in_doc = set()
            for chunk in spacy_doc.noun_chunks:
                if not self._is_valid_chunk(chunk):
                    continue
                tokens = [token.lemma_.lower() for token in chunk if not (token.is_stop or token.is_punct or token.is_space)]
                if not tokens:
                    continue
                if not (self._min_ngram <= len(tokens) <= self._max_ngram):
                    continue
                phrase = " ".join(tokens)
                term_counts[phrase] += 1
                if phrase not in seen_in_doc:
                    doc_counts[phrase][doc_index] += 1
                    seen_in_doc.add(phrase)

        candidates: List[Candidate] = []
        for phrase, total_count in term_counts.items():
            document_frequency = len(doc_counts[phrase])
            if document_frequency < self._min_document_frequency:
                continue
            candidates.append(Candidate(text=phrase, term_frequency=total_count, document_frequency=document_frequency))

        candidates.sort(key=lambda candidate: (-candidate.term_frequency, candidate.text))
        return candidates


def _deduplicate_candidates(phrases: Sequence[str], embeddings: np.ndarray, *, threshold: float = 0.9) -> List[int]:
    """Return indices of non-duplicate candidates using cosine similarity."""

    if len(phrases) <= 1:
        return list(range(len(phrases)))

    keep_indices: List[int] = []
    normalised = _normalise_vectors(embeddings)
    for index, phrase in enumerate(phrases):
        if not keep_indices:
            keep_indices.append(index)
            continue
        similarities = normalised[index] @ normalised[keep_indices].T
        if float(np.max(similarities)) >= threshold:
            LOGGER.debug("Dropping duplicate candidate '%s' (similarity %.3f)", phrase, float(np.max(similarities)))
            continue
        keep_indices.append(index)
    return keep_indices


def _normalise_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=float)
    min_value = float(values.min())
    max_value = float(values.max())
    if math.isclose(max_value, min_value):
        return {key: 1.0 for key in scores}
    scale = max_value - min_value
    return {key: (value - min_value) / scale for key, value in scores.items()}


def _compute_tfidf_scores(candidates: Sequence[Candidate], total_documents: int) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for candidate in candidates:
        tf = candidate.term_frequency
        df = candidate.document_frequency
        idf = math.log((1 + total_documents) / (1 + df)) + 1.0
        scores[candidate.text] = tf * idf
    return _normalise_scores(scores)


def _compute_similarity_scores(phrases: Sequence[str], phrase_embeddings: np.ndarray, centroid: np.ndarray) -> Dict[str, float]:
    if not phrases:
        return {}
    centroid_norm = centroid / (np.linalg.norm(centroid) or 1.0)
    similarities = _normalise_vectors(phrase_embeddings) @ centroid_norm
    score_map = {phrase: float(similarity) for phrase, similarity in zip(phrases, similarities)}
    return _normalise_scores(score_map)


def _mmr(
    phrases: Sequence[str],
    relevance_scores: Dict[str, float],
    embeddings: np.ndarray,
    *,
    diversity: float = 0.7,
    top_k: int = 5,
) -> List[int]:
    if not phrases:
        return []

    lambda_value = float(np.clip(diversity, 0.0, 1.0))
    normalised = _normalise_vectors(embeddings)
    pairwise_sim = normalised @ normalised.T
    selected: List[int] = []
    remaining = list(range(len(phrases)))

    while remaining and len(selected) < top_k:
        if not selected:
            next_index = max(remaining, key=lambda idx: relevance_scores.get(phrases[idx], 0.0))
            selected.append(next_index)
            remaining.remove(next_index)
            continue

        best_index = None
        best_score = -np.inf
        for idx in remaining:
            relevance = relevance_scores.get(phrases[idx], 0.0)
            redundancy = max(pairwise_sim[idx, selected])
            mmr_score = lambda_value * relevance - (1.0 - lambda_value) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = idx
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
    return selected


def _maxsum_rerank(candidate_indices: List[int], embeddings: np.ndarray, *, top_k: int) -> List[int]:
    if len(candidate_indices) <= top_k:
        return candidate_indices

    normalised = _normalise_vectors(embeddings[candidate_indices])
    similarity = normalised @ normalised.T

    best_subset = candidate_indices[:top_k]
    best_score = -np.inf

    from itertools import combinations

    for subset in combinations(range(len(candidate_indices)), top_k):
        subset_indices = [candidate_indices[i] for i in subset]
        subset_similarity = similarity[np.ix_(subset, subset)]
        score = float(np.sum(subset_similarity))
        if score > best_score:
            best_score = score
            best_subset = subset_indices
    return best_subset


def _compute_cohesion_score(doc_embeddings: np.ndarray) -> float:
    if len(doc_embeddings) <= 1:
        return 1.0
    normalised = _normalise_vectors(doc_embeddings)
    similarity = normalised @ normalised.T
    tri_upper = similarity[np.triu_indices_from(similarity, k=1)]
    if tri_upper.size == 0:
        return 1.0
    return float(np.mean(tri_upper))


def label_cluster(
    documents: Sequence[str],
    *,
    embedder: Optional[EmbeddingBackend] = None,
    candidate_generator: Optional[KeyphraseCandidateGenerator] = None,
    top_k: int = 5,
    diversity: float = 0.7,
    use_maxsum: bool = False,
) -> Dict[str, object]:
    """Label a cluster of documents using the configured pipeline.

    Parameters
    ----------
    documents:
        Sequence of raw document texts that form the cluster.
    embedder:
        Optional embedding backend. When ``None`` a :class:`Specter2Embedder`
        instance is created lazily.
    candidate_generator:
        Optional keyphrase candidate generator. Defaults to
        :class:`NounPhraseCandidateGenerator`.
    top_k:
        Number of keyphrases to return.
    diversity:
        Diversity term for MMR selection (``0.0`` = relevance only,
        ``1.0`` = maximum diversity).
    use_maxsum:
        When ``True`` apply a MaxSum re-ranking stage after MMR selection.

    Returns
    -------
    dict
        A JSON-serialisable dictionary containing the top phrases, the best
        phrase, and auxiliary score information.
    """

    if not documents:
        return {"top_phrases": [], "best_phrase": "", "scores": {"cohesion": 0.0}}

    embedder = embedder or Specter2Embedder()
    candidate_generator = candidate_generator or NounPhraseCandidateGenerator()

    doc_embeddings = embedder.embed(documents)
    if doc_embeddings.size == 0:
        return {"top_phrases": [], "best_phrase": "", "scores": {"cohesion": 0.0}}

    candidates = candidate_generator.generate(documents)
    if not candidates:
        cohesion = _compute_cohesion_score(doc_embeddings)
        return {"top_phrases": [], "best_phrase": "", "scores": {"cohesion": cohesion}}

    phrases = [candidate.text for candidate in candidates]
    phrase_embeddings = embedder.embed(phrases)
    keep_indices = _deduplicate_candidates(phrases, phrase_embeddings, threshold=0.9)
    phrases = [phrases[idx] for idx in keep_indices]
    phrase_embeddings = phrase_embeddings[keep_indices]
    kept_candidates = [candidates[idx] for idx in keep_indices]

    centroid = np.mean(doc_embeddings, axis=0)
    tfidf_scores = _compute_tfidf_scores(kept_candidates, len(documents))
    similarity_scores = _compute_similarity_scores(phrases, phrase_embeddings, centroid)

    combined_scores = {}
    for phrase in phrases:
        combined_scores[phrase] = 0.3 * tfidf_scores.get(phrase, 0.0) + 0.7 * similarity_scores.get(phrase, 0.0)

    selected_indices = _mmr(phrases, combined_scores, phrase_embeddings, diversity=diversity, top_k=top_k)
    if use_maxsum and selected_indices:
        selected_indices = _maxsum_rerank(selected_indices, phrase_embeddings, top_k=min(top_k, len(selected_indices)))

    ranked_phrases = sorted(selected_indices, key=lambda idx: combined_scores.get(phrases[idx], 0.0), reverse=True)
    top_phrases = [phrases[idx] for idx in ranked_phrases]

    cohesion = _compute_cohesion_score(doc_embeddings)
    return {
        "top_phrases": top_phrases,
        "best_phrase": top_phrases[0] if top_phrases else "",
        "scores": {
            "cohesion": cohesion,
            "combined": {phrase: combined_scores[phrase] for phrase in top_phrases},
            "tfidf": {phrase: tfidf_scores.get(phrase, 0.0) for phrase in top_phrases},
            "similarity": {phrase: similarity_scores.get(phrase, 0.0) for phrase in top_phrases},
        },
    }


__all__ = [
    "Candidate",
    "EmbeddingBackend",
    "KeyphraseCandidateGenerator",
    "NounPhraseCandidateGenerator",
    "Specter2Embedder",
    "label_cluster",
]

