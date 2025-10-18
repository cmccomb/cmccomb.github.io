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

   The build step loads the `sentence-transformers/all-MiniLM-L6-v2` embedding
   model for KeyBERT and the `allenai/specter2_base` model for the cluster
   labeler. Ensure both models are available locally before running the script
   in offline environments:

   ```bash
   python - <<'PY'
   from sentence_transformers import SentenceTransformer

   SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
   from transformers import AutoModel, AutoTokenizer

   AutoTokenizer.from_pretrained("allenai/specter2_base")
   AutoModel.from_pretrained("allenai/specter2_base")
   PY
   ```

   The generated `assets/json/pubs.json` file now includes a top-level
   `clusters` collection containing HDBSCAN cluster centroids and KeyBERT labels
   alongside the existing per-publication records. Each cluster is capped at
   `12.5%` of the publications to keep the visualization balanced.

## Cluster labels

The publication visualization overlays concise cluster labels at the centroid
of each HDBSCAN cluster. The positions update every render (including window
resize events), and the typography scales responsively to remain legible on
small and large displays alike.

Cluster labels are now produced by `_scripts/cluster_labeling.py`, which uses
SPECTER2 embeddings, spaCy noun-chunk candidates, and a composite scoring model
to select descriptive phrases. The module exposes a ``label_cluster`` helper:

```python
from _scripts.cluster_labeling import label_cluster

cluster_documents = ["Deep learning for medical imaging", "MRI synthesis"]
labels = label_cluster(cluster_documents, top_k=3)
print(labels["top_phrases"])
```

The return value is a JSON-friendly dictionary containing the best phrase,
selected phrases, and score breakdowns. See the module docstring for a full
description of the algorithm and configuration hooks.

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

Python unit tests exercise the data processing helpers:

```bash
pytest
```
