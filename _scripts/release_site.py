"""Validate site release metadata and publish a release after deployment."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"


def release_metadata(root: Path) -> tuple[str, str]:
    """Read the version and its dated changelog entry from the same checkout."""
    metadata = json.loads((root / "_data/release.json").read_text(encoding="utf-8"))
    version = metadata.get("version", "")
    updated = metadata.get("last_updated", "")
    if not isinstance(version, str) or not re.fullmatch(VERSION_PATTERN, version):
        raise ValueError("Site version must use MAJOR.MINOR.PATCH without leading zeros")
    if not isinstance(updated, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", updated):
        raise ValueError("last_updated must use YYYY-MM-DD")
    date.fromisoformat(updated)

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = list(re.finditer(rf"^## \[{re.escape(version)}\] - (.+)$", changelog, re.M))
    if len(headings) != 1 or headings[0].group(1) != updated:
        raise ValueError("Changelog must contain one version heading matching last_updated")
    notes = re.split(r"^## ", changelog[headings[0].end():], maxsplit=1, flags=re.M)[0].strip()
    if not notes:
        raise ValueError("The current release must have changelog notes")
    return version, notes


def github_json(endpoint: str, *, missing_ok: bool = False) -> dict | None:
    """Distinguish a missing resource from an authentication or network failure."""
    result = subprocess.run(["gh", "api", endpoint], capture_output=True, text=True)
    if result.returncode:
        if missing_ok and "(HTTP 404)" in result.stderr:
            return None
        raise RuntimeError(f"GitHub request failed for {endpoint}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def publish_release(root: Path, repository: str, commit: str, assets: tuple[Path, ...] = ()) -> None:
    """Create one release per version, without changing existing tags/releases."""
    version, notes = release_metadata(root)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Repository must use owner/name")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Release target must be the full deployed commit SHA")
    for asset in assets:
        if not asset.is_file() or not asset.stat().st_size:
            raise ValueError(f"Release asset is missing or empty: {asset}")
    tag = f"v{version}"
    prefix = f"repos/{repository}"
    existing = github_json(f"{prefix}/releases/tags/{tag}", missing_ok=True)
    if existing is not None:
        if existing.get("draft"):
            raise RuntimeError(f"{tag} already has a draft release; review it before publishing")
        print(f"{tag} already has a release; leaving it unchanged")
        return

    reference = github_json(f"{prefix}/git/ref/tags/{tag}", missing_ok=True)
    if reference is not None:
        target = reference["object"]
        for _ in range(10):
            if target["type"] != "tag":
                break
            target = github_json(f"{prefix}/git/tags/{target['sha']}")["object"]
        if target["type"] != "commit" or target["sha"] != commit:
            raise RuntimeError(f"{tag} already points elsewhere; refusing to move it")

    with tempfile.TemporaryDirectory(prefix="site-release-") as directory:
        notes_path = Path(directory) / "notes.md"
        notes_path.write_text(notes + "\n", encoding="utf-8")
        subprocess.run([
            "gh", "release", "create", tag, *[str(asset.resolve()) for asset in assets],
            "--repo", repository,
            "--target", commit, "--title", tag, "--notes-file", str(notes_path),
        ], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without contacting GitHub")
    parser.add_argument("--repository")
    parser.add_argument("--commit")
    parser.add_argument("--asset", type=Path, action="append", default=[], help="Attach a file to a new release; repeatable")
    args = parser.parse_args()
    if args.check:
        version, _ = release_metadata(ROOT)
        print(f"Validated site release v{version}")
    else:
        if not args.repository or not args.commit:
            parser.error("publishing requires --repository and --commit")
        publish_release(ROOT, args.repository, args.commit, tuple(args.asset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
