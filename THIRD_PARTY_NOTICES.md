# Third-party assets

The site keeps the following browser dependencies in the repository so page
rendering does not depend on third-party CDNs at runtime.

- Bootstrap CSS 5.1.0 — MIT License — `assets/vendor/bootstrap/LICENSE`
- D3.js 7.9.0 — ISC License — `assets/vendor/d3/LICENSE`
- Inline interface icons are adapted from Font Awesome Free 7.3.1 — CC BY 4.0
  — Copyright 2026 Fonticons, Inc.

Source releases:

- <https://github.com/twbs/bootstrap/releases/tag/v5.1.0>
- <https://github.com/d3/d3/releases/tag/v7.9.0>
- <https://github.com/FortAwesome/Font-Awesome/releases/tag/7.3.1>
- <https://creativecommons.org/licenses/by/4.0/>

## Map export typography

`_scripts/fonts/Inter.ttf` is Inter's variable font from Google Fonts commit
`0b58fb370093f9a9f4ff785d94405710b79de67c`, used only at build time for map exports.
Copyright 2020 The Inter Project Authors. Licensed under the SIL Open Font
License 1.1; the complete license is in `_scripts/fonts/OFL.txt` and accompanies
the downloadable export bundle. Exported SVG text is outlined.

- [Pinned font source](https://github.com/google/fonts/tree/0b58fb370093f9a9f4ff785d94405710b79de67c/ofl/inter)
- [Inter project](https://github.com/rsms/inter)

The build dependencies `@resvg/resvg-js` (MPL-2.0) and `fflate` (MIT) are pinned
in `package-lock.json`; they are not served as browser scripts.
