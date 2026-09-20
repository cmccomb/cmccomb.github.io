# Publication map exports

[Repository overview](../README.md) · [Development](development.md) ·
[Publication data](publication-data.md) · [Maintenance](maintenance.md)

The [download gallery](https://cmccomb.com/assets/maps/) offers ready-to-use maps
for social profiles, posts, and slides. Each format has dark and light themes,
a native-resolution PNG, a 2× PNG, and an SVG with outlined text. The ZIP contains
all images, an offline gallery, usage notes, licenses, and a provenance manifest.

## Formats

| Format | Native PNG | 2× PNG |
| --- | --- | --- |
| LinkedIn profile banner, 4:1 | 1584 × 396 | 3168 × 792 |
| Social landscape | 1200 × 627 | 2400 × 1254 |
| Social square, 1:1 | 1080 × 1080 | 2160 × 2160 |
| Social portrait, 4:5 | 1080 × 1350 | 2160 × 2700 |
| Slides, 16:9 | 1920 × 1080 | 3840 × 2160 |
| Slides, 4:3 | 1600 × 1200 | 3200 × 2400 |

Use the native PNG for social uploads and the 2× PNG or SVG for slides. Preserve
the aspect ratio. SVG text is converted to paths so it renders without installing
a font, and the SVG contains a title and description for accessible embedding.

The LinkedIn banner reserves the lower-left 23% of width and 44% of height for
the profile-photo overlay. This is a design allowance, not a platform guarantee:
check the upload preview because crops and overlays vary across devices.
The landscape format follows LinkedIn's
[1200 × 627 sharing-image guidance](https://www.linkedin.com/help/linkedin/answer/a521928).

## Generate locally

With Node.js 24, from the repository root:

```bash
npm ci
npm run export:maps
```

Open `_map_exports/index.html`, or use the files directly. The directory is
ignored by Git and excluded from Jekyll by its underscore prefix. To choose
another output directory:

```bash
npm run export:maps -- --output /path/to/map-exports
```

The command replaces its own named files without deleting the output directory.
It validates every layout before writing output. A crowded future snapshot
causes a build failure if labels or papers cannot fit; adjust the layout or add
space instead of suppressing that check or silently omitting records.

To preview the gallery through the local site, first build Jekyll and then add
the exports, since a Jekyll rebuild clears generated files:

```bash
bundle exec jekyll build --strict_front_matter
npm run export:maps -- --output _site/assets/maps
bundle exec jekyll serve --skip-initial-build --no-watch
```

Visit <http://127.0.0.1:4000/assets/maps/>. The regular `jekyll serve` command
previews the core site; the gallery needs the additional export step.

## Data and layout

[`map_exports.cjs`](../_scripts/map_exports.cjs) contains the presets, themes,
layout, validation, and SVG renderer. It uses the committed `pubs.json`, vendored
D3 7.9.0, and the live map's shared topic-label formatter. It never fetches data,
queries Scholar, or recomputes the embedding. Refresh publication records using
the separate [publication-data workflow](publication-data.md).

Every publication remains a circle. Color uses the same Magma year scale as the
website; radius scales with the square root of citation count with a minimum
visible size for uncited papers. The size legend uses exactly the same radii.
The existing embedding is fitted to each aspect ratio; deterministic collision
avoidance keeps circles, topic labels, legends, and the avatar allowance clear.
Absolute distances between papers are not a quantitative metric.

[`export_publication_maps.cjs`](../_scripts/export_publication_maps.cjs) writes
PNGs with the pinned resvg renderer, outlines SVG text using the vendored Inter
font, and builds the gallery, manifest, and ZIP. No browser or system fonts are
required. The manifest includes the snapshot SHA-256, dataset commit and build
date, site version, and each image's size and SHA-256. The release date describes
the site version; it does not imply the publication data was refreshed that day.
The same inputs and locked runtime produce deterministic outputs.

## Deployment and releases

The deployment build compiles into `_site/assets/maps/` after Jekyll. Every
successful deployment updates the gallery and its `publication-maps.zip` from
the exact tested commit. The ZIP is also preserved as a workflow artifact and
passed to the release job, which attaches it when publishing a new site version.
Existing versions and older releases are left unchanged.
Successful CI runs also provide a `publication-map-previews` artifact containing
the full ZIP, so reviewers can inspect images and the offline gallery before merge.

Thus the gallery represents the latest deployed snapshot, while a GitHub Release
preserves its version's bundle. A publication refresh merged through the normal
workflow regenerates the gallery automatically; it does not need a separate
export commit. Follow the normal version-bump policy for data/site changes.

## Validation

```bash
npm run test:unit
npm run test:browser -- tests/browser/map-downloads.spec.ts
```

Unit checks cover all presets, record/topic retention, clipping and overlap,
determinism, output dimensions, checksums, archive contents, and gallery links.
Browser checks compile the gallery, check accessibility and mobile overflow,
and download the ZIP. Visually inspect representative native PNGs after changing
the layout or font; geometry assertions cannot judge composition or readability.
