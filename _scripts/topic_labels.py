"""Shared normalization and basic checks for automatically derived phrases."""

from __future__ import annotations

import re


BAD_START_WORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "is", "of", "on", "or", "our", "to", "via", "we", "with",
})
BAD_END_WORDS = BAD_START_WORDS | {"the", "this", "these", "those"}


def normalise_topic_label(label: str) -> str:
    """Use one comparison key for generated and committed topic labels."""

    return " ".join(re.findall(r"[a-z0-9]+", label.casefold()))


def suitable_label(label: str) -> bool:
    """Reject incomplete phrases such as ``market in`` and ``work we``."""

    words = normalise_topic_label(label).split()
    return (
        len(words) >= 2
        and words[0] not in BAD_START_WORDS
        and words[-1] not in BAD_END_WORDS
    )
