"""Tests for the committed publication snapshot validator."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from _scripts.validate_publications_json import (
    SnapshotValidationError,
    validate_payload,
)


def valid_payload() -> dict[str, object]:
    """Return a minimal valid publication snapshot."""

    return {
        "meta": {
            "record_count": 1,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "records": [
            {
                "author_pub_id": "author:publication",
                "x": 1.0,
                "y": 2.0,
                "num_citations": 3,
                "bib_dict": {"title": "A publication"},
            }
        ],
        "clusters": [{"id": 0, "label": "design research"}],
    }


def test_valid_payload_passes() -> None:
    """A structurally complete snapshot should validate."""

    validate_payload(valid_payload(), max_age_days=1)


def test_duplicate_publication_ids_fail() -> None:
    """Duplicate nodes would create ambiguous graph links and must fail CI."""

    payload = valid_payload()
    payload["records"] = payload["records"] * 2  # type: ignore[operator]
    payload["meta"]["record_count"] = 2  # type: ignore[index]

    with pytest.raises(SnapshotValidationError, match="Duplicate publication id"):
        validate_payload(payload)


def test_nonfinite_coordinate_fails() -> None:
    """NaN coordinates should be rejected before browser deployment."""

    payload = valid_payload()
    payload["records"][0]["x"] = float("nan")  # type: ignore[index]

    with pytest.raises(SnapshotValidationError, match="invalid x coordinate"):
        validate_payload(payload)
