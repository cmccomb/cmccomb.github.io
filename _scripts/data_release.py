"""Advance and verify site releases when publication data changes."""

from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SITE_TIMEZONE = ZoneInfo("America/New_York")
VERSION_PATTERN = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")


def _version(value: str) -> tuple[int, int, int]:
    match = VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid site version: {value}")
    return tuple(int(part) for part in match.groups())


def _release(path: Path) -> dict[str, str]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Release metadata must be an object")
    version = metadata.get("version")
    updated = metadata.get("last_updated")
    if not isinstance(version, str) or not isinstance(updated, str):
        raise ValueError("Release metadata is missing its version or date")
    _version(version)
    date.fromisoformat(updated)
    return metadata


def _snapshot(root: Path) -> dict:
    payload = json.loads((root / "assets/json/pubs.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("meta"), dict):
        raise ValueError("Publication snapshot metadata is missing")
    return payload


def _snapshot_date(payload: dict) -> date:
    built_at = payload["meta"].get("built_at_utc")
    if not isinstance(built_at, str):
        raise ValueError("Publication snapshot build time is missing")
    built = datetime.fromisoformat(built_at.replace("Z", "+00:00"))
    if built.tzinfo is None:
        raise ValueError("Publication snapshot build time has no timezone")
    return built.astimezone(SITE_TIMEZONE).date()


def prepare_data_release(root: Path = ROOT, *, as_of: date | None = None) -> str:
    """Bump the patch release in the same branch as a refreshed graph."""

    payload = _snapshot(root)
    release_path = root / "_data/release.json"
    metadata = _release(release_path)
    release_date = as_of or datetime.now(SITE_TIMEZONE).date()
    if release_date < _snapshot_date(payload):
        raise ValueError("Release date precedes the publication snapshot")
    major, minor, patch = _version(metadata["version"])
    new_version = f"{major}.{minor}.{patch + 1}"
    metadata["version"] = new_version
    metadata["last_updated"] = release_date.isoformat()

    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Publication snapshot has no records")
    source_commit = payload["meta"].get("dataset_commit")
    if isinstance(source_commit, str) and re.fullmatch(r"[0-9a-f]{40}", source_commit):
        source = f"dataset commit `{source_commit[:12]}`"
    else:
        source = "the current source dataset"
    entry = (
        f"## [{new_version}] - {release_date.isoformat()}\n\n"
        "### Data\n\n"
        f"- Refresh the publication graph with {len(records)} publications from {source}.\n\n"
    )
    changelog_path = root / "CHANGELOG.md"
    changelog = changelog_path.read_text(encoding="utf-8")
    first_release = re.search(r"^## \[", changelog, flags=re.MULTILINE)
    if first_release is None:
        raise ValueError("Changelog has no release entry")
    changelog_path.write_text(
        changelog[:first_release.start()] + entry + changelog[first_release.start():],
        encoding="utf-8",
    )
    release_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return new_version


def check_data_release(base: str, root: Path = ROOT) -> bool:
    """Require a newer release in every commit that changes publication JSON."""

    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise ValueError("Base must be a full Git commit SHA")
    diff = subprocess.run(
        ["git", "diff", "--quiet", base, "HEAD", "--", "assets/json/pubs.json"],
        cwd=root,
        check=False,
    )
    if diff.returncode == 0:
        return False
    if diff.returncode != 1:
        raise RuntimeError("Could not compare publication snapshots")

    previous = subprocess.run(
        ["git", "show", f"{base}:_data/release.json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    old_release = json.loads(previous.stdout)
    new_release = _release(root / "_data/release.json")
    if _version(new_release["version"]) <= _version(old_release["version"]):
        raise ValueError("Publication data changed without a new site version")
    if date.fromisoformat(new_release["last_updated"]) < _snapshot_date(_snapshot(root)):
        raise ValueError("Site update date precedes the publication refresh")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    check = commands.add_parser("check")
    check.add_argument("--base", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(f"Prepared publication data release v{prepare_data_release()}")
    elif check_data_release(args.base):
        print("Publication data change includes a new site release")
    else:
        print("Publication data is unchanged")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
