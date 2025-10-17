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
   python3 _scripts/build_json.py
   ```

   The build step loads the `sentence-transformers/all-MiniLM-L6-v2` embedding
   model for KeyBERT. Ensure the model assets are available locally before
   running the script in offline environments:

   ```bash
   python - <<'PY'
   from sentence_transformers import SentenceTransformer

   SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
   PY
   ```

   The generated `assets/json/pubs.json` file now includes a top-level
   `clusters` collection containing DBSCAN cluster centroids and KeyBERT labels
   alongside the existing per-publication records.

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
