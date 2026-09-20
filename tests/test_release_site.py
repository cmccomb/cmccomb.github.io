"""Check that release publishing preserves version history and exact commits."""

import json
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest

from _scripts import release_site


COMMIT = "a" * 40


@pytest.fixture
def release_root(tmp_path):
    (tmp_path / "_data").mkdir()
    (tmp_path / "_data/release.json").write_text(json.dumps({
        "version": "0.1.0", "last_updated": "2026-09-20",
    }))
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [0.1.0] - 2026-09-20\n\nCurrent notes.\n\n"
        "## [0.0.1] - 2026-09-19\n\nOld notes.\n"
    )
    return tmp_path


def test_metadata_extracts_only_current_release(release_root):
    assert release_site.release_metadata(release_root) == ("0.1.0", "Current notes.")


@pytest.mark.parametrize("version,updated", [
    ("0.1", "2026-09-20"),
    ("00.1.0", "2026-09-20"),
    ("0.1.0", "2026-02-30"),
    ("0.1.0", "2026-09-21"),
])
def test_invalid_or_inconsistent_metadata_is_rejected(release_root, version, updated):
    (release_root / "_data/release.json").write_text(json.dumps({
        "version": version, "last_updated": updated,
    }))
    with pytest.raises(ValueError):
        release_site.release_metadata(release_root)


def test_new_release_targets_deployed_commit_and_uses_changelog(release_root, monkeypatch):
    monkeypatch.setattr(release_site, "github_json", Mock(return_value=None))
    calls = []

    def create(args, **kwargs):
        calls.append(args)
        assert args[args.index("--target") + 1] == COMMIT
        assert args[args.index("--repo") + 1] == "example/site"
        assert Path(args[args.index("--notes-file") + 1]).read_text() == "Current notes.\n"
        assert kwargs["check"] is True

    monkeypatch.setattr(release_site.subprocess, "run", create)
    release_site.publish_release(release_root, "example/site", COMMIT)
    assert len(calls) == 1
    assert calls[0][:4] == ["gh", "release", "create", "v0.1.0"]


def test_existing_release_is_unchanged_on_later_deploy(release_root, monkeypatch):
    api = Mock(return_value={"draft": False})
    mutation = Mock()
    monkeypatch.setattr(release_site, "github_json", api)
    monkeypatch.setattr(release_site.subprocess, "run", mutation)
    release_site.publish_release(release_root, "example/site", COMMIT)
    assert api.call_count == 1
    mutation.assert_not_called()


def test_existing_draft_is_not_silently_treated_as_published(release_root, monkeypatch):
    monkeypatch.setattr(release_site, "github_json", Mock(return_value={"draft": True}))
    with pytest.raises(RuntimeError, match="draft release"):
        release_site.publish_release(release_root, "example/site", COMMIT)


def test_conflicting_tag_is_preserved(release_root, monkeypatch):
    monkeypatch.setattr(release_site, "github_json", Mock(side_effect=[
        None, {"object": {"type": "commit", "sha": "b" * 40}},
    ]))
    mutation = Mock()
    monkeypatch.setattr(release_site.subprocess, "run", mutation)
    with pytest.raises(RuntimeError, match="refusing to move"):
        release_site.publish_release(release_root, "example/site", COMMIT)
    mutation.assert_not_called()


def test_matching_annotated_tag_can_receive_release(release_root, monkeypatch):
    monkeypatch.setattr(release_site, "github_json", Mock(side_effect=[
        None,
        {"object": {"type": "tag", "sha": "b" * 40}},
        {"object": {"type": "commit", "sha": COMMIT}},
    ]))
    mutation = Mock()
    monkeypatch.setattr(release_site.subprocess, "run", mutation)
    release_site.publish_release(release_root, "example/site", COMMIT)
    mutation.assert_called_once()


@pytest.mark.parametrize("status", [401, 403, 500])
def test_api_errors_do_not_trigger_release_creation(monkeypatch, status):
    monkeypatch.setattr(release_site.subprocess, "run", Mock(return_value=
        subprocess.CompletedProcess([], 1, "", f"gh: request failed (HTTP {status})")))
    with pytest.raises(RuntimeError, match="GitHub request failed"):
        release_site.github_json("repos/example/site/releases/tags/v0.1.0", missing_ok=True)


def test_missing_release_is_distinct_from_api_failure(monkeypatch):
    monkeypatch.setattr(release_site.subprocess, "run", Mock(return_value=
        subprocess.CompletedProcess([], 1, "", "gh: Not Found (HTTP 404)")))
    assert release_site.github_json("repos/example/site/releases/tags/v0.1.0", missing_ok=True) is None
