# Changelog

Site releases use `MAJOR.MINOR.PATCH` versions, beginning with `0.1.0`.
Earlier tags and releases remain available in GitHub history.

## [0.2.8] - 2026-10-29

### Improved

- Make publication topic labels smaller and subtler, place them clear of map
  circles, and display “Vision + language” for the vision-language topic.

## [0.2.7] - 2026-10-29

### Fixed

- Resolve duplicate publication topic labels automatically during graph refreshes.

### Data

- Refresh the publication graph with 267 publications from dataset commit `0359eb8a8a1c`.

## [0.2.6] - 2026-10-22

### Fixed

- Space the citation legend circles evenly between their outlines, including
  when the map resizes.

## [0.2.5] - 2026-10-22

### Fixed

- Give the mixed design-methods publication group a distinct reviewed topic,
  leaving “additive manufacturing” on the group where it is more representative.
- Reject duplicate topic names during graph generation and snapshot validation
  so a future refresh requires review before publication.

### Improved

- Show each publication's title and year in map tooltips without repeating its
  topic; keep topics available in map labels, accessible names, list, details,
  and search.
- Make highly cited publications modestly larger in the interactive map and
  downloadable figures while preserving the existing minimum circle size.

## [0.2.4] - 2026-10-02

### Data

- Publish the October quarterly graph refresh with 262 publications.
- Replace the incomplete “market in” topic with the reviewed “design decisions” label.

### Maintenance

- Advance the site release with future publication data refreshes, and require
  a new version whenever the committed graph changes.
- Reject incomplete generated topic phrases before they reach the site.
- Load publication data with a release-versioned URL so refreshed pages bypass
  cached copies of an older graph.
- Refresh the pinned Python and GitHub Actions dependencies.

## [0.2.3] - 2026-09-20

### Improved

- Replace the favicon with a bold charcoal C on a diagonal mauve-to-peach
  gradient, matching the publication map palette.
- Use the icon on the homepage, 404 page, and map download gallery, with
  versioned URLs so new releases refresh the cached favicon.

## [0.2.2] - 2026-09-20

### Improved

- Give the publication search field and selected List/Map control a muted
  mauve-to-peach gradient, with readable dark text and a matching focus ring.

## [0.2.1] - 2026-09-20

### Improved

- Show the search toolbar in the blurred List background while keeping the
  preview inert and hidden from assistive technology.
- Remove the dot between the site version and update date in the shared footer
  and map download gallery.

## [0.2.0] - 2026-09-20

### Added

- Automatically compile publication maps for LinkedIn profile banners, landscape,
  square and portrait social posts, and 16:9 and 4:3 slides.
- Include dark and light themes, native and 2× PNGs, and scalable SVGs with
  outlined typography. Each layout preserves every paper and topic, avoids
  overlaps, and uses the live map's year palette and topic labels.
- Publish a [map download gallery](https://cmccomb.com/assets/maps/) and ZIP bundle
  on each deployment; attach that deployment's bundle to each new GitHub Release.
- Record snapshot provenance, dimensions, and checksums alongside the exports,
  with automated geometry, bundle, and gallery accessibility checks.

## [0.1.1] - 2026-09-20

### Maintenance

- Update Playwright to 1.63.0 and axe-core's Playwright integration to 4.13.0.
- Update NumPy to 2.5.2 and setuptools to 84.0.0.
- Move Python tooling to 3.12, satisfying NumPy's minimum version, and use one
  `.python-version` file for CI, publication refreshes, and release publishing.

## [0.1.0] - 2026-09-20

First release using three-part site versioning. This establishes the current
site as the baseline and includes the recent publication-browser improvements.

### Added

- Searchable publication List and Map views with shareable search and paper URLs.
- Publication details, source links, citation copying, and keyboard navigation.
- Development, architecture, publication-data, and maintenance guides, plus
  contribution instructions and a pull request template.
- An `llms.txt` guide to the bio, CV, and structured publication data.
- Site version and last-updated date in the footer, with GitHub Releases
  created after successful deployment.

### Improved

- Responsive toolbar, map, list, and detail layouts across narrow and short screens.
- Colorful list previews that match the selected view behind the profile.
- Background clipping during resize, CV icon contrast, and quieter footer text.
- Browser accessibility and regression coverage, validated publication snapshots,
  and deployment of the exact revision that passed CI.
