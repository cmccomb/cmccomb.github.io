# cmccomb.github.io

1. Site layout with [Jekyll](https://jekyllrb.com/)
2. Basic formatting with [Bootstrap](https://getbootstrap.com/)
3. Publication visualization with [D3.js](https://d3js.org/)
4. Icons by Font Awesome (no changes made, [see license](https://fontawesome.com/license))

## Setup

1. Install Ruby dependencies:

   ```bash
   bundle install
   ```

2. Install Python dependencies and build JSON:

   ```bash
   pip install -r _scripts/requirements.txt
   python3 _scripts/build_json.py --dataset ccm/publications --revision main --seed 42
   ```

   The CLI is deterministic: the seed is recorded in the generated
   `assets/json/pubs.json` file alongside the configured K-means settings and
   projection perplexity. Toggle verbosity or dry-run behaviour as required:

   ```bash
   python3 _scripts/build_json.py --dry-run --verbose
   python3 _scripts/build_json.py --force --seed 2025
   ```

   Offline environments are supported by pre-populating the Hugging Face cache
   and setting `HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. The script will
   reuse cached copies of the `allenai/specter2` base model and adapter without
   hitting the network.

   Cluster labels now come from class-based TF-IDF summaries with KeyBERT MMR
   fallback, and clustering happens in PCA-reduced space for improved
   stability. Default K-means clustering (eight centroids) keeps the topics
   balanced without any additional hyperparameter tuning.

## Cluster labels

The publication visualization overlays concise cluster labels at the centroid
of each K-means cluster. The positions update every render (including window
resize events), and the typography scales responsively to remain legible on
small and large displays alike.

3. Serve the site locally:

   ```bash
   bundle exec jekyll serve
   ```

## Tests

Run the automated test suite to validate the helpers and ensure deterministic
builds:

```bash
pytest
```

You can still compile the static site locally with:

```bash
bundle exec jekyll build
```
