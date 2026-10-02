"""Editorial topic labels and basic checks for generated phrases."""

from __future__ import annotations

import re
from collections.abc import Iterable


# Exact membership keeps a reviewed label from following a re-formed cluster.
CURATED_LABELS = {
    frozenset({
        "0P9w_S0AAAAJ:70eg2SAEIzsC",
        "0P9w_S0AAAAJ:BrmTIyaxlBUC",
        "0P9w_S0AAAAJ:K3LRdlH-MEoC",
        "0P9w_S0AAAAJ:OU6Ihb5iCvQC",
        "0P9w_S0AAAAJ:RGFaLdJalmkC",
        "0P9w_S0AAAAJ:SP6oXDckpogC",
        "0P9w_S0AAAAJ:XiVPGOgt02cC",
        "0P9w_S0AAAAJ:bEWYMUwI8FkC",
        "0P9w_S0AAAAJ:bz8QjSJIRt4C",
        "0P9w_S0AAAAJ:tOudhMTPpwUC",
        "0P9w_S0AAAAJ:yD5IFk8b50cC",
    }): "design decisions",
}

BAD_START_WORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "is", "of", "on", "or", "our", "to", "via", "we", "with",
})
BAD_END_WORDS = BAD_START_WORDS | {"the", "this", "these", "those"}


def curated_label(publication_ids: Iterable[str]) -> str | None:
    """Return a human-reviewed label for an unchanged cluster membership."""

    return CURATED_LABELS.get(frozenset(publication_ids))


def suitable_label(label: str) -> bool:
    """Reject incomplete phrases such as ``market in`` and ``work we``."""

    words = re.findall(r"[a-z0-9]+", label.lower())
    return (
        len(words) >= 2
        and words[0] not in BAD_START_WORDS
        and words[-1] not in BAD_END_WORDS
    )
