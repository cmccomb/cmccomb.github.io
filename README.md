# cmccomb.com

Source for [cmccomb.com](https://cmccomb.com/), a single-page Jekyll profile
with an interactive D3 publication map.

## Architecture

- Jekyll renders the profile and GitHub Pages hosts the static output.
- Bootstrap CSS and D3 are stored under `assets/vendor` so the live page does
  not depend on third-party CDNs.
- `assets/json/pubs.json` is a committed, validated snapshot of the
  [`ccm/publications`](https://huggingface.co/datasets/ccm/publications)
  Hugging Face dataset.
- `_scripts/build_json.py` converts existing publication embeddings into the
  two-dimensional map and labels clusters with coverage-aware class-based
  TF-IDF.
- The separate
  [`scrape-my-publications`](https://github.com/cmccomb/scrape-my-publications)
  repository refreshes Scholar metadata and embeddings on the first day of
  January, April, July, and October. This repository rebuilds the graph on the
  following day and opens or updates a pull request from the controlled
  `automation/publication-graph-refresh` branch. Merging that pull request uses
  the normal tested deployment path.

PyTorch and model downloads are not needed in this repository: the graph build
uses embeddings already stored in the publication dataset.

## Local development

Install the locked Ruby dependencies and build the site:

```bash
bundle install
bundle exec jekyll build --strict_front_matter
bundle exec jekyll serve
```

Install the pinned Python test dependencies:

```bash
python -m pip install --requirement _scripts/requirements-dev.txt
python -m pytest
python _scripts/validate_publications_json.py
```

Install the locked browser test dependencies and Chromium:

```bash
npm ci
npx playwright install chromium
npm run test:browser
```

Regenerate the graph from the current dataset:

```bash
python _scripts/build_json.py --dataset ccm/publications --revision main --seed 42
python _scripts/validate_publications_json.py --max-age-days 1
```

The generated JSON records the requested dataset revision and its resolved
Hugging Face commit SHA for provenance.

## Automation

- `CI` runs the Jekyll build, Python tests, snapshot validation, JavaScript
  syntax checks, and Playwright/axe browser accessibility tests with read-only
  permissions.
- `Deploy site` publishes only the exact `master` push revision that passed CI;
  it has no manual deployment path.
- `Refresh publication graph` runs quarterly with read-only build permissions,
  validates the proposed snapshot, and passes it to a narrowly permissioned job
  that updates a pull request. It never pushes to `master` or deploys.
- Dependabot monitors Ruby, Python, npm, and GitHub Actions dependencies.

All external Actions are pinned to immutable commit SHAs. The live page also
uses a restrictive content security policy and renders publication metadata as
text rather than HTML.
