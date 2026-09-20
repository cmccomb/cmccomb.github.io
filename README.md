# cmccomb.com

Source for [Chris McComb's personal site](https://cmccomb.com/): a Jekyll
profile with a searchable publication list and an interactive D3 map.
GitHub Pages serves static output; the browser reads a committed publication
snapshot, so visiting the site does not query Scholar or run a model.

## Run locally

Use Ruby 3.3 and Bundler 2.6, then run from the repository root:

```bash
bundle install
bundle exec jekyll serve
```

Open <http://127.0.0.1:4000>. The committed publication data is enough to preview
the core site. Node.js also compiles the map downloads; Python handles data work.
See [Development](docs/development.md) for setup, checks, and troubleshooting.

## Documentation

| Guide | What it covers |
| --- | --- |
| [Development](docs/development.md) | Runtime versions, local preview, test commands, failure reports |
| [Architecture](docs/architecture.md) | Directory map, rendering, browser state, responsive behavior |
| [Publication data](docs/publication-data.md) | Sources, JSON structure, provenance, graph regeneration |
| [Map exports](docs/map-exports.md) | Social and slide formats, local compilation, gallery, release bundles |
| [Maintenance](docs/maintenance.md) | Routine edits, dependency updates, deployment, rollback |
| [Changelog](CHANGELOG.md) | Version history and GitHub release notes |
| [Contributing](CONTRIBUTING.md) | Change and review workflow |
| [Security](SECURITY.md) | Private vulnerability reporting |

## Where to make changes

| Change | Source |
| --- | --- |
| Bio | [index.md](index.md) |
| Site version and last-updated date | [_data/release.json](_data/release.json) and [CHANGELOG.md](CHANGELOG.md) |
| Name, affiliation, profile links, CV path, metadata | [_config.yml](_config.yml) |
| Profile and publication markup | [_layouts/home.html](_layouts/home.html) |
| Profile styles / publication styles | [default_style.css](assets/css/default_style.css) / [home_style.css](assets/css/home_style.css) |
| Search, citations, paper links | [publication_helpers.js](assets/js/publication_helpers.js) |
| List, map, and publication details | [graph_layout.js](assets/js/graph_layout.js) |
| Profile navigation and view preference | [interface.js](assets/js/interface.js) |
| Publication snapshot | [Data workflow](docs/publication-data.md), then [pubs.json](assets/json/pubs.json) |
| Downloadable map formats and themes | [map_exports.cjs](_scripts/map_exports.cjs); [live gallery](https://cmccomb.com/assets/maps/) |
| AI-readable site guide | [llms.txt](llms.txt), published at [/llms.txt](https://cmccomb.com/llms.txt) |

## Delivery and licensing

Pull requests run the site build, publication-data checks, and browser tests.
A successful CI run for a push to `master` triggers deployment of that exact
commit. Publication refreshes propose changes through pull requests.
After deployment, a new site version receives a GitHub Release with its changelog
notes and a map-download ZIP; an existing version's release stays unchanged.
See [Maintenance](docs/maintenance.md) for the full release path.

The repository uses the [MIT license](LICENSE.md). Vendored assets, export fonts,
and icon attributions are listed in [Third-party notices](THIRD_PARTY_NOTICES.md).
