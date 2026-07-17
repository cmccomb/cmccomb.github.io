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
  each month. This repository rebuilds and deploys the graph on the second day.

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

Regenerate the graph from the current dataset:

```bash
python _scripts/build_json.py --dataset ccm/publications --revision main --seed 42
python _scripts/validate_publications_json.py --max-age-days 1
```

The generated JSON records the requested dataset revision and its resolved
Hugging Face commit SHA for provenance.

## Automation

- `CI` runs the Jekyll build, Python tests, snapshot validation, and JavaScript
  syntax checks with read-only permissions.
- `Deploy site` publishes only the exact `master` revision that passed CI.
- `Refresh publication graph` rebuilds the committed snapshot monthly,
  validates it, records the update, and deploys it in the same trusted run.
- Dependabot monitors Ruby, Python, and GitHub Actions dependencies.

All external Actions are pinned to immutable commit SHAs. The live page also
uses a restrictive content security policy and renders publication metadata as
text rather than HTML.
