# cmccomb.github.io

1. Site layout with [Jekyll](https://jekyllrb.com/)
2. Basic formatting with [Bootstrap](https://getbootstrap.com/)
3. Publication visualization with [D3.js](https://d3js.org/)
4. Icons by Font Awesome (no changes made, [see license](https://fontawesome.com/license))
5. Responsive cluster labels summarizing research areas directly on the scatter plot

## Setup

1. Install Ruby dependencies:

   ```bash
   bundle install
   ```

2. Install Python dependencies and build JSON:

   ```bash
   pip install -r _scripts/requirements.txt
   python3 _scripts/build_json.py
   ```

   The build step loads the `allenai/specter2` adapter on top of the
   `allenai/specter2_base` transformer to generate KeyBERT embeddings.
   Ensure both the base model and adapter weights are available locally before
   running the script in offline environments:

   ```bash
   python - <<'PY'
   from _scripts import build_json

   build_json.ensure_sentence_transformer(build_json.KEYBERT_MODEL_NAME)
   PY
   ```

   The generated `assets/json/pubs.json` file now includes a top-level
   `clusters` collection containing DBSCAN cluster centroids and KeyBERT labels
   alongside the existing per-publication records. Each cluster is capped at
   `12.5%` of the publications to keep the visualization balanced.

## Cluster labels

The publication visualization overlays concise cluster labels at the centroid
of each DBSCAN cluster. The positions update every render (including window
resize events), and the typography scales responsively to remain legible on
small and large displays alike.

3. Serve the site locally:

   ```bash
   bundle exec jekyll serve
   ```

## Tests

Run a local build to verify the site compiles:

```bash
bundle exec jekyll build
```

Deployment to GitHub Pages runs only after this check passes on `master`.
