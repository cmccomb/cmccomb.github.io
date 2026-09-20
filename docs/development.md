# Development

[Repository overview](../README.md) · [Architecture](architecture.md) ·
[Publication data](publication-data.md) · [Maintenance](maintenance.md)

Run commands below from the repository root. The versions used in CI are
Ruby 3.3 (from [`.ruby-version`](../.ruby-version)), Node.js 24, and Python 3.12
(from [`.python-version`](../.python-version)).
The lockfile was generated with Bundler 2.6.9; use Bundler 2.6 for the documented setup.
Use the committed lockfiles and pinned Python requirements.

## Preview the site

```bash
bundle install
bundle exec jekyll serve
```

Open <http://127.0.0.1:4000>. The server rebuilds when source files change;
restart it after editing `_config.yml`. The committed `assets/json/pubs.json`
supports the full publication browser without installing Python or regenerating
the graph. `_scripts/serve.sh` is a convenience wrapper for the same server.

To use another port:

```bash
bundle exec jekyll serve --host 127.0.0.1 --port 4001
```

For a production-style build without a server:

```bash
JEKYLL_ENV=production bundle exec jekyll build --strict_front_matter
```

Output is written to `_site/`. Edit the source, not that directory.

## Browser and JavaScript checks

Install Node.js 24, then:

```bash
npm ci
npx playwright install chromium
npm run test:unit
npm run test:browser
```

On a Linux machine that also needs browser system libraries, use
`npx playwright install --with-deps chromium` for the browser installation step.
Playwright performs a strict production build, compiles the map downloads, and starts its own server at
`127.0.0.1:4173`. Keep that port free; the suite intentionally does not reuse an
existing server. Ruby and Bundler must be available in the same shell.

Useful focused runs:

```bash
npm run test:browser -- tests/browser/site.spec.ts
npm run test:browser -- tests/browser/publication-browser.spec.ts
npm run test:browser -- tests/browser/responsive-layout.spec.ts --workers=2
npm run test:browser:headed
```

`npm test` runs the browser suite; run `npm run test:unit` separately for the
JavaScript helpers and map exports. Local failures retain screenshots and traces in
`test-results/`. CI also uploads an HTML `playwright-report` artifact on failure.
After downloading and extracting that artifact, open it with
`npx playwright show-report playwright-report`.

## Python checks

Create a Python 3.12 environment for graph development and tests. NumPy 2.5
requires Python 3.12 or newer; recreate an existing 3.11 environment before
installing the updated requirements.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --requirement _scripts/requirements-dev.txt
python -m pytest
python _scripts/validate_publications_json.py
python _scripts/release_site.py --check
```

The validator uses the Python standard library. Running it alone does not
require the graph dependencies. Tests use fixtures; routine tests and website
builds do not refresh the upstream dataset. For an intentional graph refresh,
follow [Publication data](publication-data.md).

The release check also uses only the standard library. It validates the version,
date, and changelog entry without contacting GitHub. Release tests mock GitHub
calls and never create tags or releases.

## Choose the relevant checks

| Change | Local validation |
| --- | --- |
| Documentation, Liquid, metadata, or `llms.txt` | Strict Jekyll build; inspect rendered output and verify links |
| Layout, styles, navigation, accessibility | Browser suite and visual inspection of affected states |
| Search, citation formatting, resource URLs | JavaScript unit tests and publication-browser tests |
| Builder, schema, or publication snapshot | Python tests, snapshot validator, and browser suite for changed data |
| Dependencies or workflows | Affected local checks plus the complete PR CI run |
| Map exports or download gallery | JavaScript unit tests, gallery browser tests, and visual inspection of PNGs; see [Map exports](map-exports.md) |

CI runs the full configured checks regardless of which local checks are selected.
For a complete local pass, use the build, JavaScript, browser, and Python commands
above. Dependency audits run in CI and are described in
[Maintenance](maintenance.md#dependencies).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Bundler rejects the Ruby version | Check `ruby --version`, `which ruby`, and `bundle --version`; activate Ruby 3.3 in the current shell |
| Configuration edits are missing in preview | Restart Jekyll after changing `_config.yml` |
| Playwright cannot start the server | Free port 4173 and confirm `bundle exec jekyll build --strict_front_matter` succeeds |
| Chromium executable is missing | Run `npx playwright install chromium` after `npm ci` |
| Publication browser reports unavailable data | Run the snapshot validator and inspect the browser console; confirm `assets/json/pubs.json` is served |
| Python packages are missing | Activate `.venv` and install `_scripts/requirements-dev.txt` |
