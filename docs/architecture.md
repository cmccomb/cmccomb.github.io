# Architecture

[Repository overview](../README.md) · [Development](development.md) ·
[Publication data](publication-data.md) · [Maintenance](maintenance.md)

The site has three stages: Python prepares a committed publication snapshot,
Jekyll builds static files, and JavaScript renders the publication browser.
There is no application server or browser-side model inference. Bootstrap CSS
and D3 are vendored, so the page does not fetch them from a CDN.

## Repository map

| Path | Responsibility |
| --- | --- |
| `index.md`, `_config.yml` | Bio and shared site identity/configuration |
| `_data/release.json`, `CHANGELOG.md` | Site version, displayed update date, and release notes |
| `_layouts/` | Homepage and 404 markup |
| `_includes/seo.html` | Canonical URLs, social metadata, Person structured data, `llms.txt` discovery |
| `_includes/footer.html` | Shared copyright, version, and update date on the homepage and 404 |
| `assets/css/` | Profile styling and responsive publication layouts |
| `assets/js/` | Navigation, publication rendering, and pure formatting/search helpers |
| `assets/json/pubs.json` | Committed publication records, clusters, and build provenance |
| `assets/images/`, `assets/files/` | Profile images, icons, and the public CV PDF |
| `assets/vendor/` | Browser dependencies and their licenses |
| `_scripts/` | Graph builder, snapshot validator, release publisher, Python requirements, and preview helper |
| `tests/` | Python fixtures/tests, JavaScript unit tests, and Playwright browser tests |
| `.github/` | CI, deployment, publication refresh, dependency updates, PR template |
| `docs/`, `CONTRIBUTING.md` | Repository guides, excluded from the built website |
| `llms.txt`, `robots.txt`, `sitemap.xml` | Public discovery files rendered by Jekyll |
| `Gemfile`, `comb.gemspec`, `Gemfile.lock` | Ruby dependencies; the gemspec retains the site's original theme packaging |

`_site/`, `.venv/`, `node_modules/`, and test reports are local generated output.
Jekyll exclusions in `_config.yml` and Git ignores have different jobs:
excluding a file from Git does not, by itself, exclude it from the built site.

## Page and browser responsibilities

[`_layouts/home.html`](../_layouts/home.html) renders the profile, publication
toolbar, map container, results list, detail panel, and accessible status regions.
Its scripts load in order: D3, shared publication helpers, graph rendering, then
profile/navigation behavior.

| Module | Owns |
| --- | --- |
| [`publication_helpers.js`](../assets/js/publication_helpers.js) | Search normalization/ranking, citation formatting, source link validation; also loaded by Node unit tests |
| [`graph_layout.js`](../assets/js/graph_layout.js) | Snapshot loading, map/list rendering, search, selection, details, clipboard behavior, legends, keyboard movement, resizing |
| [`interface.js`](../assets/js/interface.js) | Profile visibility, navigation history, focus restoration, Escape behavior, responsive default and remembered view |

The graph module reads `assets/json/pubs.json` once per page load. It validates
records for rendering and provides an unavailable-data state with a Scholar
link if loading fails. Publication metadata is inserted as text. The homepage
also offers a Scholar link when JavaScript is disabled.

## State and navigation

The profile is the root URL. Any `view`, `q`, or `paper` query parameter opens
the publication browser:

| Parameter | Meaning |
| --- | --- |
| `view=list` or `view=map` | Selected presentation |
| `q` | Search text, normalized for matching |
| `paper` | Stable `author_pub_id` for a selected publication |

For example, `/?view=list&q=machine%20learning` is a shareable search.
Search normalizes accents, punctuation, and AI terminology; short terms match
whole words and longer terms can match word prefixes. Title matches rank first.
With no query, the list sorts newest first. Switching views preserves search
and selection. Reloads and Back/Forward navigation restore state from the URL.

`interface.js` dispatches `publicationgraph:visibilitychange` when opening or
closing the browser. `graph_layout.js` dispatches `publicationgraph:viewchange`
when the user chooses a view. The two modules share the current view through
`data-publication-view` on `#graph-container`.

Before an explicit choice, widths up to 768 px default to List; larger viewports
default to Map. An explicit choice is remembered during the current page visit,
including a return to the profile. It is not stored across visits. The blurred
profile background uses the selected view and remains inert, hidden from
assistive technology, and clipped to prevent background scrollbars.

## Layout and accessibility

The map uses a minimum 1024 × 720 canvas so papers and labels remain legible.
Smaller viewports scroll within the active map. Year colors use D3's Magma scale;
circle sizes represent citation counts. List accents reuse the year colors.

The toolbar's measured height reserves the content area below it. List/detail
columns share its centered width, and long details scroll below a fixed Close
control. Keep these relationships intact when changing toolbar content.

Publication controls support keyboard use, visible focus, accessible names,
status announcements, and focus restoration. Arrow keys move between map
papers. Escape closes details first, then clears search, then returns to the
profile. Playwright/axe checks and responsive regression tests cover these
behaviors; visual inspection remains part of layout work.
