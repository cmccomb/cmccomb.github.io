# Changelog

Site releases use `MAJOR.MINOR.PATCH` versions, beginning with `0.1.0`.
Earlier tags and releases remain available in GitHub history.

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
