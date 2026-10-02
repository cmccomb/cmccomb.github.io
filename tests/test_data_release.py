"""Check the publication refresh and release coupling."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import subprocess

import pytest

from _scripts.data_release import check_data_release, prepare_data_release


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def test_data_change_requires_and_prepares_new_release(tmp_path: Path) -> None:
    """A graph-only commit must fail the gate; a dated patch release passes."""

    (tmp_path / "assets/json").mkdir(parents=True)
    (tmp_path / "_data").mkdir()
    snapshot_path = tmp_path / "assets/json/pubs.json"
    snapshot = {
        "meta": {
            "built_at_utc": "2026-10-02T13:58:19+00:00",
            "dataset_commit": "a" * 40,
        },
        "records": [{"author_pub_id": "one"}],
        "clusters": [{"id": 0, "label": "design decisions"}],
    }
    snapshot_path.write_text(json.dumps(snapshot) + "\n", encoding="utf-8")
    (tmp_path / "_data/release.json").write_text(
        json.dumps({"version": "0.2.3", "last_updated": "2026-09-20"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [0.2.3] - 2026-09-20\n\nPrior release.\n",
        encoding="utf-8",
    )
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-qm", "previous")
    base = _git(tmp_path, "rev-parse", "HEAD")

    snapshot["records"].append({"author_pub_id": "two"})
    snapshot_path.write_text(json.dumps(snapshot) + "\n", encoding="utf-8")
    _git(tmp_path, "add", "assets/json/pubs.json")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-qm", "refresh data")
    with pytest.raises(ValueError, match="without a new site version"):
        check_data_release(base, tmp_path)

    assert prepare_data_release(tmp_path, as_of=date(2026, 10, 2)) == "0.2.4"
    _git(tmp_path, "add", "_data/release.json", "CHANGELOG.md")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-qm", "release data")
    assert check_data_release(base, tmp_path)
    assert "## [0.2.4] - 2026-10-02" in (
        tmp_path / "CHANGELOG.md"
    ).read_text(encoding="utf-8")
