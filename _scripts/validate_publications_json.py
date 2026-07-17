"""Validate the committed publication graph snapshot before deployment."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SnapshotValidationError(ValueError):
    """Raised when the publication snapshot is unsafe or incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SnapshotValidationError(message)


def validate_payload(payload: Any, *, max_age_days: int | None = None) -> None:
    """Validate graph schema, record integrity, and optional snapshot freshness."""

    _require(isinstance(payload, dict), "Snapshot must be a JSON object")
    records = payload.get("records")
    clusters = payload.get("clusters")
    metadata = payload.get("meta")

    _require(isinstance(records, list) and records, "Snapshot has no records")
    _require(isinstance(clusters, list) and clusters, "Snapshot has no clusters")
    _require(isinstance(metadata, dict), "Snapshot metadata is missing")
    _require(
        metadata.get("record_count") == len(records),
        "Metadata record count does not match snapshot",
    )

    publication_ids: set[str] = set()
    for index, record in enumerate(records):
        _require(isinstance(record, dict), f"Record {index} is not an object")
        publication_id = record.get("author_pub_id")
        _require(
            isinstance(publication_id, str) and publication_id,
            f"Record {index} has no publication id",
        )
        _require(
            publication_id not in publication_ids,
            f"Duplicate publication id: {publication_id}",
        )
        publication_ids.add(publication_id)

        for coordinate in ("x", "y"):
            value = record.get(coordinate)
            _require(
                isinstance(value, (int, float)) and math.isfinite(value),
                f"Record {index} has an invalid {coordinate} coordinate",
            )

        citations = record.get("num_citations")
        _require(
            isinstance(citations, (int, float)) and citations >= 0,
            f"Record {index} has an invalid citation count",
        )
        bibliography = record.get("bib_dict")
        _require(
            isinstance(bibliography, dict)
            and isinstance(bibliography.get("title"), str)
            and bool(bibliography.get("title")),
            f"Record {index} has no title",
        )

    if max_age_days is not None:
        built_at = metadata.get("built_at_utc")
        _require(isinstance(built_at, str), "Snapshot build time is missing")
        built_time = datetime.fromisoformat(built_at.replace("Z", "+00:00"))
        _require(built_time.tzinfo is not None, "Snapshot build time has no timezone")
        age = datetime.now(timezone.utc) - built_time.astimezone(timezone.utc)
        _require(age.days <= max_age_days, "Snapshot is older than allowed")


def main() -> int:
    """Load and validate a publication snapshot from disk."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("assets/json/pubs.json"),
    )
    parser.add_argument("--max-age-days", type=int)
    args = parser.parse_args()

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    validate_payload(payload, max_age_days=args.max_age_days)
    print(f"Validated {len(payload['records'])} publication records")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
