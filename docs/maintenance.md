# Maintenance

[Repository overview](../README.md) · [Development](development.md) ·
[Architecture](architecture.md) · [Publication data](publication-data.md)

## Routine content edits

Update the bio in `index.md` and identity, affiliation, contact links, headshot
path, and CV path in `_config.yml`. Replace the public CV PDF at its existing
path to preserve incoming links. The repository contains the published PDF;
it is not the CV authoring source.

After identity or link changes, inspect the profile, Person structured data,
social metadata, and generated `llms.txt`. Keep `CNAME` and `site.url` aligned if
the domain changes. The [README's change map](../README.md#where-to-make-changes)
identifies other editing locations.

## AI-readable content

[`llms.txt`](../llms.txt) renders to `/llms.txt` without an HTML layout. It follows
the [llms.txt proposal](https://llmstxt.org/): a short site summary and curated
links to the profile, CV, publication JSON, and related authoritative sources.
The homepage advertises it with `rel="describedby"`.

The file derives identity from `_config.yml` and research interests from the
homepage content. Keep interpretation notes and resource links in the template;
avoid duplicating publication records, citation counts, or build timestamps.
The JSON's `meta` object carries snapshot provenance. This guide supplements
`robots.txt`, the sitemap, and structured data. It does not control crawler
access or guarantee that a tool will use it.

Repository documentation belongs in `docs/` and is explicitly excluded from
Jekyll output, along with `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, and policy files.
Check `_site/` after adding another top-level maintenance file so it is not
published accidentally.

## Workflows

| Workflow | Trigger and responsibility |
| --- | --- |
| [CI](../.github/workflows/ci.yml) | Pull requests, pushes to `master`, or manual dispatch; read-only build/test/audit jobs |
| [Deploy site](../.github/workflows/deploy.yml) | Successful CI for a `master` push in this repository; builds and deploys the exact tested SHA, then publishes a new version's release |
| [Refresh publication graph](../.github/workflows/refresh-publication-graph.yml) | Quarterly or manual on `master`; read-only build, then a narrowly permissioned proposal job |
| [Dependabot](../.github/dependabot.yml) | Monthly Ruby, Python, npm, and GitHub Actions update proposals |

The refresh proposal job updates its controlled automation branch, opens or
updates the PR, and explicitly dispatches CI for that branch. Review its data
diff as described in [Publication data](publication-data.md). Merging it follows
the same release path as a normal code change.

## Release and verify

1. Open a PR against `master` and wait for `site`, `publication-data`, and
   `Browser accessibility` to pass. Review the exact revision to be merged.
2. Merge the PR. The resulting `master` commit runs CI again.
3. Wait for `Deploy site`, including its release job, to succeed for that same commit. A manual CI run or a
   successful PR run alone does not trigger deployment.
4. Verify the affected content at <https://cmccomb.com/>. For browser changes,
   check the relevant profile, List, Map, search, and detail states. For
   `llms.txt`, fetch the deployed text and verify its links and rendered values.
   For a new version, also verify the release notes and tag's target commit on
   [GitHub Releases](https://github.com/cmccomb/cmccomb.github.io/releases), including
   its map ZIP. For map changes, inspect the [download gallery](https://cmccomb.com/assets/maps/).

The [Actions history](https://github.com/cmccomb/cmccomb.github.io/actions)
records CI and deployment results. Match the deployed workflow's head SHA to
the merge commit when reporting a release as live. Pages is configured to use
GitHub Actions; the deploy job uses the `github-pages` environment.

To roll back a regression, make a revert commit through a PR and follow the
same checks and deployment path. This preserves history and the guarantee that
the deployed revision was tested. The deployment workflow has no manual bypass.

## Site versions

[`_data/release.json`](../_data/release.json) is the source of truth for the
site version and the footer's last-updated date. It is independent of the
historical theme gem version and the private npm test-package version.
The date describes the site release, not the build time or dataset scrape time.

Use stable `MAJOR.MINOR.PATCH` values without a `v` prefix in this file. Tags,
GitHub Releases, and the footer add the prefix. While the site is below 1.0,
use a minor bump for a substantial feature or redesign and a patch bump for
fixes and content/data updates. Repository-only documentation changes may keep
the current site version and date.

For each new site version:

1. Set `version` and `last_updated` (`YYYY-MM-DD`) in `_data/release.json`.
2. Add one matching `## [VERSION] - YYYY-MM-DD` entry to `CHANGELOG.md`, describing
   the changes since the prior version. Keep older entries intact.
3. Run `python _scripts/release_site.py --check` and the relevant site checks.
4. Follow the release and verification steps above.

After Pages deploys successfully, its release job checks out the same tested
commit and runs [`release_site.py`](../_scripts/release_site.py). A version
without a release gets a GitHub Release and tag targeting that commit. Its notes
come only from the matching changelog entry. The publisher also attaches the
map ZIP produced by that deployment's build job; it does not regenerate the
archive in the release job. See [Map exports](map-exports.md). A later deployment with the same
version leaves the existing release unchanged. Older tags/releases, including
the legacy `v0.1`, remain untouched.

The publisher refuses to move a conflicting tag and reports an existing draft
instead of treating it as published. Authentication/network errors fail the
job. If publication fails after Pages deploys, the site may already be live;
fix the cause and rerun the failed release job for the same workflow run.
The `--check` mode is read-only; invoking the script with `--repository` and
`--commit` publishes and is normally reserved for the deployment job.

## Dependencies

Ruby dependencies are locked in `Gemfile.lock`, browser test and map export dependencies in
`package-lock.json`, and Python versions in `_scripts/requirements*.txt`.
The supported Python runtime lives in `.python-version`; CI, graph refreshes,
and release publishing all read that file.
External Actions use immutable SHAs. Keep release comments next to those SHAs
accurate when updating them.

CI runs `bundle-audit check --update`, `python -m pip_audit --local`, and
`npm audit --audit-level=moderate`. The Ruby audit tool is installed by CI;
Python's audit tool is in the development requirements.

Vendored Bootstrap CSS and D3 do not receive npm updates: update their files
and licenses deliberately, adjust references if filenames change, and update
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). Run browser tests and
inspect the UI after changing browser assets. Keep the existing content
security policy compatible with the assets the site actually serves.
