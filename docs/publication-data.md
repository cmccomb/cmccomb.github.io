# Publication data

[Repository overview](../README.md) · [Development](development.md) ·
[Architecture](architecture.md) · [Maintenance](maintenance.md)

## Source and ownership

[`scrape-my-publications`](https://github.com/cmccomb/scrape-my-publications)
maintains Scholar metadata and embeddings in the
[`ccm/publications`](https://huggingface.co/datasets/ccm/publications) dataset.
This repository converts that dataset into
[`assets/json/pubs.json`](../assets/json/pubs.json), the snapshot used by the site.
Jekyll and browser tests use the committed snapshot; they do not scrape Scholar.

The upstream refresh runs on the first day of January, April, July, and October.
This repo's [refresh workflow](../.github/workflows/refresh-publication-graph.yml)
runs at 03:23 America/New_York on the following day and can also be dispatched
manually on `master`. It proposes the generated JSON through
`automation/publication-graph-refresh`; it does not deploy directly.

Correct bibliographic source errors upstream so later refreshes retain the fix.
Keep stable Scholar `author_pub_id` values: the site uses them in shared paper
URLs. Topic labels are generated summaries and may change when the collection
changes.

## Build process

[`build_json.py`](../_scripts/build_json.py) reads the dataset's `train` split.
It requires uniform `embedding` arrays, `bib_dict`, `author_pub_id`, and
`num_citations`. Existing embeddings supply the input; no PyTorch installation
or model download is needed.

The builder projects embeddings using t-SNE followed by PCA, clusters with
K-means in reduced embedding space, refines clusters in projection space, and
labels them with coverage-aware class-based TF-IDF. It writes the snapshot
atomically. The browser adjusts positions for its viewport and avoids label
collisions; stored coordinates are the input to that layout.

## Snapshot structure

| Field | Contents and use |
| --- | --- |
| `meta` | Build timestamp, source dataset/revision, resolved commit, random seed, algorithm settings, library versions, record count |
| `records` | Publication metadata and map coordinates |
| `clusters` | Topic identifiers, labels, and centroids |

Each record contains `author_pub_id`, `bib_dict`, `pub_year`, `num_citations`,
`x`, `y`, and `cluster_id`. `bib_dict` holds the title and available author,
abstract, journal/conference, volume, pages, and citation metadata. Optional
`pub_url`, `eprint_url`, and `doi` values are retained when present in the source.
A cluster contains `id`, `label`, and `centroid` with `x` and `y` coordinates.

Citation counts are snapshot values. `meta.built_at_utc` dates the graph build,
not a new Scholar scrape. `meta.dataset_revision` records the requested ref;
`meta.dataset_commit` records the resolved SHA when resolution succeeds. If
resolution fails or runs offline, the builder may retain the requested ref
instead of an immutable SHA. Check that field before claiming exact provenance.

Paper links come from source data. Helpers also recognize DOIs explicitly
embedded in known publisher URL formats. Missing links fall back to Scholar;
a publisher link does not establish open access. The helper tests cover URL
validation and link labeling.

## Regenerate intentionally

Set up the Python environment using [Development](development.md#python-checks).
Then run:

```bash
python _scripts/build_json.py --dataset ccm/publications --revision main --seed 42
python _scripts/validate_publications_json.py --max-age-days 1
python -m pytest
```

Use `--revision <dataset-commit-sha>` instead of `main` to build from a specific
dataset version. Use `--dry-run` to run the pipeline without writing the JSON,
and `--verbose` for detailed logs. These commands may download the dataset.
The build timestamp changes between runs, even with the same source and seed,
so do not expect a byte-identical JSON file.

Before committing the snapshot, inspect its record count, source commit,
available links, topic labels, and visual map/list behavior. Run the browser
suite when publishing changed data. Commit the generated JSON alongside any
builder/schema changes so the site and tests use the same contract.

## Validation and freshness

[`validate_publications_json.py`](../_scripts/validate_publications_json.py)
checks the top-level structure, matching record count, unique nonempty IDs,
finite coordinates, nonnegative citation counts, and nonempty titles.
`--max-age-days` additionally checks the build timestamp. It does not verify
the correctness of each bibliographic claim or availability of external URLs.

Normal CI validates the committed snapshot without an age limit. The refresh
workflow requires a build no more than one day old. A successful routine site
build therefore does not imply that Scholar metadata has just been refreshed.
