"""Derive complete, distinctive topic phrases from publication evidence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.metadata import version
import logging
import re
from typing import Mapping, Sequence

import numpy
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfTransformer
from snowballstemmer import stemmer

if __package__:
    from .topic_labels import normalise_topic_label, suitable_label
else:  # pragma: no cover - direct script execution
    from topic_labels import normalise_topic_label, suitable_label

LOGGER = logging.getLogger(__name__)
TITLE_WEIGHT = 2
MIN_COVERAGE = 0.20
COMPLETION_RETENTION = 0.80
MAX_WORDS = 5
TOKEN = re.compile(r"\b[^\W\d_]+(?:[-‐‑–][^\W\d_]+)*\b", re.UNICODE)
EDGES = ENGLISH_STOP_WORDS | {"using", "based", "including", "toward", "towards", "within", "via"}
STEMMER = stemmer("english")


def labeling_metadata() -> dict:
    """Describe the reproducible title derivation used by a snapshot."""

    return {
        "algorithm": "paper-supported-ctfidf-v2",
        "source_fields": ["title", "abstract"],
        "title_weight": TITLE_WEIGHT,
        "max_phrase_words": MAX_WORDS,
        "minimum_paper_fraction": MIN_COVERAGE,
        "completion_retention": COMPLETION_RETENTION,
        "stemmer": f"snowballstemmer-{version('snowballstemmer')}",
    }


@dataclass
class Evidence:
    titles: set[int] = field(default_factory=set)
    abstracts: set[int] = field(default_factory=set)
    surfaces: dict[str, "Evidence"] = field(default_factory=dict)

    def papers(self, members: set[int]) -> set[int]:
        return (self.titles | self.abstracts) & members


@lru_cache(maxsize=32768)
def _stem(token: str) -> str:
    return "-".join(STEMMER.stemWord(part) for part in token.split("-"))


def _phrases(text: str):
    # Sentence/parenthesis boundaries keep phrases contiguous and exclude
    # appended acronyms such as "models (VLMs)" from a term's wording.
    for sentence in re.split(r"[.!?;:()\n]+", text):
        words = TOKEN.findall(sentence.lower().replace("‐", "-").replace("‑", "-").replace("–", "-"))
        for start in range(len(words)):
            for length in range(1, min(MAX_WORDS, len(words) - start) + 1):
                phrase = words[start:start + length]
                if length == 1 and "-" not in phrase[0]:
                    continue
                surface = " ".join(phrase)
                if phrase[0] in EDGES or phrase[-1] in EDGES or not suitable_label(surface):
                    continue
                yield " ".join(_stem(word) for word in phrase), surface


def complete_terms(support: dict[str, set[int]]) -> set[str]:
    """Keep complete phrases without cascading into poorly supported extensions."""

    complete = []
    for key in sorted(support, key=lambda value: (-len(value.split()), value)):
        if not any(
            f" {key} " in f" {longer} "
            and len(support[longer]) >= len(support[key]) * COMPLETION_RETENTION
            for longer in complete
        ):
            complete.append(key)
    return set(complete)


def assign_unique_labels(candidates: dict[int, list[tuple[str, float]]]) -> dict[int, str]:
    """Keep distinct titles while minimizing each cluster's relative score loss."""

    def concept(label: str) -> str:
        return " ".join(_stem(word) for word in normalise_topic_label(label).split())

    keys = sorted({concept(label) for options in candidates.values() for label, _ in options})
    groups = sorted(candidates)
    if len(keys) < len(groups):
        raise ValueError("No distinct supported topic labels for all clusters")
    columns = {key: index for index, key in enumerate(keys)}
    costs = numpy.full((len(groups), len(keys)), numpy.inf)
    surfaces = {}
    for row, group in enumerate(groups):
        options = candidates[group]
        maximum = max((score for _, score in options), default=0)
        if maximum <= 0:
            raise ValueError(f"Cluster {group} has no supported topic label")
        for rank, (label, score) in enumerate(options):
            key = concept(label)
            column = columns[key]
            cost = -score / maximum + rank / (len(options) + 1) * 1e-9
            if cost < costs[row, column]:
                costs[row, column] = cost
                surfaces[group, key] = label
    try:
        rows, selected = linear_sum_assignment(costs)
    except ValueError as exc:
        raise ValueError("No distinct supported topic labels for all clusters") from exc
    if not numpy.isfinite(costs[rows, selected]).all():
        raise ValueError("No distinct supported topic labels for all clusters")
    return {groups[row]: surfaces[groups[row], keys[column]] for row, column in zip(rows, selected)}


def derive_labels(records: Sequence[Mapping], labels: Sequence[int]) -> dict[int, str]:
    """Rank source phrases by paper support and distinctiveness across clusters."""

    if len(records) != len(labels):
        raise ValueError("Every publication must have a cluster label")
    groups = sorted(set(labels) - {-1})
    if not groups:
        return {}
    evidence: dict[str, Evidence] = defaultdict(Evidence)
    for index, (record, group) in enumerate(zip(records, labels)):
        if group == -1:
            continue
        bib = record.get("bib_dict") or {}
        for source, attr in (("title", "titles"), ("abstract", "abstracts")):
            text = bib.get(source)
            if not isinstance(text, str):
                continue
            for key, surface in set(_phrases(text)):
                term = evidence[key]
                getattr(term, attr).add(index)
                variant = term.surfaces.setdefault(surface, Evidence())
                getattr(variant, attr).add(index)
    vocabulary = sorted(evidence)
    if not vocabulary:
        raise ValueError("No supported topic phrases in publication metadata")
    members = {group: {index for index, value in enumerate(labels) if value == group} for group in groups}
    counts = numpy.array([
        [TITLE_WEIGHT * len(evidence[key].titles & members[group])
         + len(evidence[key].abstracts & members[group]) for key in vocabulary]
        for group in groups
    ])
    scores = TfidfTransformer(norm=None).fit_transform(counts).toarray()
    candidates = {}
    for row, group in enumerate(groups):
        papers = members[group]
        minimum = min(len(papers), max(2, int(numpy.ceil(len(papers) * MIN_COVERAGE))))
        support = {key: evidence[key].papers(papers) for key in vocabulary}
        eligible = {key for key in vocabulary if len(support[key]) >= minimum}
        # Prefer the complete term over a nested fragment when it retains
        # almost all the supporting papers, including singular/plural variants.
        complete = complete_terms({key: support[key] for key in eligible})
        ranked = []
        for column, key in enumerate(vocabulary):
            if key not in complete:
                continue
            term = evidence[key]
            surface = min(term.surfaces, key=lambda value: (
                -len(term.surfaces[value].papers(papers)),
                -len(term.surfaces[value].titles & papers), len(value), value,
            ))
            ranked.append((surface, float(scores[row, column]), len(support[key]), len(term.titles & papers)))
        ranked.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))
        candidates[group] = [(surface, score) for surface, score, _, _ in ranked]
    chosen = assign_unique_labels(candidates)
    for group, title in chosen.items():
        term = next(item for item in candidates[group] if item[0] == title)
        LOGGER.info("Cluster %s: %s (score %.2f)", group, title, term[1])
    return chosen
