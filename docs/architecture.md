# Architecture

[Repository overview](../README.md) · [Development](development.md) ·
[Publication data](publication-data.md) · [Maintenance](maintenance.md)

The site has three stages: Python prepares a committed publication snapshot,
Jekyll builds static files, and JavaScript renders the publication browser.
There is no application server or browser-side model inference. Bootstrap CSS
and D3 are vendored, so the page does not fetch them from a CDN.
During deployment, Node also compiles the publication snapshot into downloadable
images and a gallery, then adds them to the static output after Jekyll.

## Repository map

| Path | Responsibility |
| --- | --- |
| `index.md`, `_config.yml` | Bio and shared site identity/configuration |
| `_data/release.json`, `CHANGELOG.md` | Site version, displayed update date, and release notes |
| `_layouts/` | Page shells for the homepage and 404 |
| `_includes/` | Profile, publication browser, shared head and footer, and SEO metadata |
| `assets/css/` | Shared and profile styles in `default_style.css`. Publication browser styles in `home_style.css`. |
| `assets/js/` | Navigation, publication rendering, and pure formatting/search helpers |
| `assets/json/pubs.json` | Committed publication records, clusters, and build provenance |
| `_plugins/` | Jekyll generators for profile assets and frontend cache versions |
| `assets/images/`, `assets/files/` | Profile images, icons, and the public CV PDF |
| `assets/vendor/` | Browser dependencies and their licenses |
| `_scripts/` | Graph builder, snapshot validator, map exports and vendored font, release publisher, Python requirements, and preview helper |
| `tests/` | Python fixtures/tests, JavaScript unit tests, and Playwright browser tests |
| `.github/` | CI, deployment, publication refresh, dependency updates, PR template |
| `docs/`, `CONTRIBUTING.md` | Repository guides, excluded from the built website |
| `llms.txt`, `robots.txt`, `sitemap.xml` | Public discovery files rendered by Jekyll |
| `Gemfile`, `comb.gemspec`, `Gemfile.lock` | Ruby dependencies; the gemspec retains the site's original theme packaging |

`_site/`, `_map_exports/`, `.venv/`, `node_modules/`, and test reports are local generated output.
Jekyll exclusions in `_config.yml` and Git ignores have different jobs:
excluding a file from Git does not, by itself, exclude it from the built site.

## Page and browser responsibilities

[`_layouts/home.html`](../_layouts/home.html) assembles the shared head and footer,
the [profile](../_includes/profile.html), and the
[publication browser](../_includes/publication-browser.html). The layout passes
the rendered Markdown bio to the profile include. The browser include owns the
toolbar, map container, results, details, and status markup.

| Asset | Source | When it loads |
| --- | --- | --- |
| Static map preview | [Profile generator](../_plugins/publication_profile.rb) projects points from `pubs.json` into `publication-preview.svg` | With the profile |
| Compact bibliography | The same generator projects citation fields into `recent-publications.json` | When the profile first opens |
| D3 and the browser controller | Vendored D3 and `graph_layout.js` | After the profile sits idle, or immediately on Explore or a publication deep link |
| Full snapshot and map renderer | Committed `pubs.json` and `publication-map.js` | In parallel when the browser controller starts |

Shared publication helpers and profile navigation load with the page. Jekyll
[hashes authored CSS and JavaScript](../_plugins/frontend_revision.rb) to version
their URLs. Preview and publication data URLs use the site release version.

| Module | Owns |
| --- | --- |
| [`publication_helpers.js`](../assets/js/publication_helpers.js) | Search normalization/ranking, topic text formatting, citation formatting, source link validation; shared with Node tests and exports |
| [`graph_layout.js`](../assets/js/graph_layout.js) | Snapshot loading and normalization, lists, search, selection, details, clipboard actions, and map keyboard and tooltip behavior |
| [`publication-map.js`](../assets/js/publication-map.js) | Map drawing, citation and year legends, collision layout, Fit map, and canvas resizing |
| [`interface.js`](../assets/js/interface.js) | Recent papers, background and on-demand loading, visibility, history, focus restoration, Escape, and preferred view |

The profile waits for the page and recent papers to finish loading, then delays
background requests by two seconds and uses the browser’s next idle period when
available. It skips these requests on connections that report Data Saver or 3G
speeds and below. An Explore action reuses any requests already underway. The
controller constructs the map and list when Explore opens. It retries a failed
background snapshot request once on entry and uses only valid records. If loading
fails on entry, navigation offers a Scholar link. Publication
metadata is inserted as text. The homepage also offers Scholar when JavaScript
is disabled.

The homepage samples up to three distinct papers from the current year when
at least six are available. Otherwise, it samples from the current and previous
years. The card resamples on a full page load. Its links open the corresponding
publication detail in List view. The card matches the profile card's height on desktop and stacks
below it on narrow screens. Explore publications appears under the citations
when the card is available and under the profile links otherwise.

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

The modules share the current view through `data-publication-view` on
`#graph-container`. Navigation sets the value before loading the browser.
The browser reads it when restoring URL state.

| Event on `#graph-container` | Sender | Effect |
| --- | --- | --- |
| `publicationgraph:visibilitychange` | Navigation | The browser initializes on its first opening and restores URL state on later changes. |
| `publicationgraph:viewchange` | Browser | Navigation remembers the List or Map choice. |
| `publicationgraph:ready` | Browser | Navigation moves focus to Search when it was waiting on Back to profile. |
| `publicationgraph:unavailable` | Browser | Navigation shows the Scholar fallback. |

Before an explicit choice, widths up to 768 px default to List; larger viewports
default to Map. An explicit choice is remembered during the current page visit,
including a return to the profile. It is not stored across visits. The static
profile backdrop is decorative. The interactive browser stays hidden and inert
while the profile is visible.

## Layout and accessibility

The map uses a minimum 1024 × 720 canvas so papers and labels remain legible.
Smaller viewports scroll within the active map. Fit map scales the full canvas
to the available space, retaining search matches and selection. At small
scales it hides topic labels to keep the overview legible. Toggling it off
restores the scrollable view and its prior position. Year colors use D3's
Magma scale, and circle sizes represent citation counts. List accents reuse
the year colors.

The toolbar's measured height reserves the content area below it. List/detail
columns share its centered width, and long details scroll below a fixed Close
control. Keep these relationships intact when changing toolbar content.

Publication controls support keyboard use, visible focus, accessible names,
status announcements, and focus restoration. Arrow keys move between map
papers. Escape closes details first, then clears search, then returns to the
profile. Playwright/axe checks and responsive regression tests cover these
behaviors; visual inspection remains part of layout work.
