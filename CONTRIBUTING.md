# Contributing

Start with the [README](README.md) to find the source for your change and the
[development guide](docs/development.md) to run the site locally.

## Change workflow

1. Create a focused branch from the current `master`.
2. Edit the source files. `_site/`, test reports, and local dependency folders
   are generated output and stay out of commits.
3. Run the checks relevant to the change using the table in
   [Development](docs/development.md#choose-the-relevant-checks).
4. Update the relevant guide when behavior, commands, data fields, or workflow
   ownership changes.
   For a site release, also update `_data/release.json` and its matching
   `CHANGELOG.md` entry using the [versioning guide](docs/maintenance.md#site-versions).
5. Open a pull request describing the problem, resulting behavior, and
   validation. Include before/after screenshots for visual changes.

All required checks must pass before merge. The production CI run and Pages
deployment are separate steps; a merged pull request alone does not establish
that the change is live. Follow [Maintenance](docs/maintenance.md#release-and-verify).

## Implementation conventions

- Keep site identity in `_config.yml` and the bio in `index.md`; `llms.txt`
  reuses both. Avoid copying changing publication counts into documentation.
- Keep deterministic search and citation logic in `publication_helpers.js`,
  separate from DOM rendering. Render dataset text through text nodes, and
  use the existing URL validation helpers for external publication links.
- Preserve keyboard navigation, focus restoration, accessible control names,
  deep links, and the profile's inert background when editing the browser.
- Check narrow and short viewports as well as a desktop view. Include empty
  search results and long publication details when changing layout.
- Update publication metadata through the documented source/build workflow.
  Preserve stable `author_pub_id` values because shared links use them.
- Keep dependency versions and external Action revisions pinned. Include
  lockfile and [third-party notice](THIRD_PARTY_NOTICES.md) updates when applicable.

Report vulnerabilities using the [security policy](SECURITY.md).
