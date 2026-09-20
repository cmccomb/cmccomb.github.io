#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { createHash } = require("node:crypto");
const { parseArgs } = require("node:util");
const { Resvg } = require("@resvg/resvg-js");
const { zipSync } = require("fflate");
const { FORMATS, THEMES, xml, layoutMap, mapSVG } = require("./map_exports.cjs");
const ROOT = path.resolve(__dirname, "..");
const sha256 = contents => createHash("sha256").update(contents).digest("hex");

function gallery(manifest) {
    return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Publication map downloads · Chris McComb</title>
<link rel="icon" type="image/svg+xml" href="../images/favicon.svg?v=${xml(manifest.version)}">
<meta name="description" content="Download Chris McComb's publication map for LinkedIn, social media, and slides in six formats and two themes.">
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f4f1;color:#212529;font:17px/1.6 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:56px 24px}h1{font-size:clamp(2rem,5vw,3.4rem);line-height:1.1;margin:20px 0}h2{margin:0;font-size:1.45rem}h3{font-size:1rem;margin:0}p{max-width:760px}a{color:#164f83;text-underline-offset:.2em}a:hover{color:#102d4b}a:focus-visible{outline:3px solid #125aba;outline-offset:5px}header{margin-bottom:48px}.eyebrow,.dimensions{color:#58616a}.eyebrow{font-size:.9rem}.download{display:inline-block;background:#212529;color:#fff;padding:12px 20px;border-radius:8px;text-decoration:none;font-weight:600}.download:hover{color:#fff;background:#394550}section{margin-top:42px}.dimensions{margin:6px 0 16px}.variants{display:grid;grid-template-columns:1fr 1fr;gap:20px}.variant{min-width:0;border:1px solid #cbd0d4;border-radius:12px;overflow:hidden;background:white}.preview{display:block;background:#e8e9e9;border-bottom:1px solid #cbd0d4}.preview img{display:block;width:100%;height:auto}.links{padding:16px 20px}.links nav{display:flex;flex-wrap:wrap;gap:8px 20px;margin-top:8px}footer{border-top:1px solid #cbd0d4;margin-top:56px;padding-top:20px;color:#58616a;font-size:.9rem}@media(max-width:700px){main{padding:32px 16px}.variants{grid-template-columns:1fr}}
</style></head><body><main>
<header><a href="/">← cmccomb.com</a><p class="eyebrow">CHRIS MCCOMB · RESEARCH</p><h1>Publication maps</h1>
<p>The same research, sized for where you share it. Choose a dark or light background, download a PNG, or use the scalable SVG in your slides.</p>
<p>${manifest.publications} publications · ${manifest.topics} topics · ${manifest.years.join("–")}</p>
<a class="download" href="publication-maps.zip" download>Download all formats · ZIP</a></header>
${FORMATS.map(format => `<section aria-labelledby="${format.id}"><h2 id="${format.id}">${xml(format.name)}</h2>
<p class="dimensions">${format.width} × ${format.height} px · 2× PNG: ${format.width * 2} × ${format.height * 2} px${format.banner ? ". Space at the lower left accommodates a profile photo; check LinkedIn’s crop preview before saving." : ""}</p>
<div class="variants">${Object.keys(THEMES).map(theme => {
        const item = manifest.images.find(i => i.format === format.id && i.theme === theme);
        const title = `${theme[0].toUpperCase() + theme.slice(1)} background`;
        return `<article class="variant"><a class="preview" href="${item.files.png.path}"><img src="${item.files.png.path}" width="${format.width}" height="${format.height}" loading="lazy" alt="${xml(format.name)} publication map on a ${theme} background"></a><div class="links"><h3>${title}</h3><nav aria-label="${xml(format.name)} ${theme} downloads"><a href="${item.files.png.path}" download>PNG</a><a href="${item.files.png2x.path}" download>PNG · 2×</a><a href="${item.files.svg.path}" download>SVG</a></nav></div></article>`;
    }).join("")}</div></section>`).join("\n")}
<footer>Site v${xml(manifest.version)} &ensp; Updated ${xml(manifest.updated)} · Publication data from ${xml(manifest.dataBuiltAt.slice(0, 10))}.
<p>Color shows publication year; circle size shows citations. Layouts adapt to each aspect ratio. <a href="README.txt">Usage and data notes</a> · <a href="manifest.json">Export details</a></p></footer>
</main></body></html>\n`;
}

function buildExports(outputDir = path.join(ROOT, "_map_exports")) {
    const snapshot = fs.readFileSync(path.join(ROOT, "assets/json/pubs.json"));
    const data = JSON.parse(snapshot);
    const release = JSON.parse(fs.readFileSync(path.join(ROOT, "_data/release.json")));
    const manifest = {
        version: release.version, updated: release.last_updated,
        publications: data.records.length, topics: data.clusters.length,
        years: [Math.min(...data.records.map(r => r.pub_year)), Math.max(...data.records.map(r => r.pub_year))],
        source: "https://cmccomb.com/assets/json/pubs.json", snapshotSha256: sha256(snapshot),
        dataBuiltAt: data.meta.built_at_utc, datasetCommit: data.meta.dataset_commit,
        images: [],
    };
    const files = {};
    const addFile = (name, contents, dimensions = {}) => {
        const buffer = Buffer.from(contents);
        files[name] = buffer;
        return { path: name, ...dimensions, bytes: buffer.length, sha256: sha256(buffer) };
    };
    for (const format of FORMATS) {
        const layout = layoutMap(data, format);
        for (const theme of Object.keys(THEMES)) {
            const stem = `${format.id}-${theme}`;
            const svg = mapSVG(layout, theme);
            const entry = { format: format.id, name: format.name, theme, width: format.width, height: format.height, files: {} };
            entry.files.svg = addFile(`${stem}.svg`, svg, { width: format.width, height: format.height });
            for (const scale of [1, 2]) {
                const renderer = new Resvg(svg, { fitTo: { mode: "zoom", value: scale }, font: { loadSystemFonts: false } });
                const image = renderer.render();
                if (image.width !== format.width * scale || image.height !== format.height * scale) throw new Error("Unexpected PNG dimensions");
                entry.files[scale === 1 ? "png" : "png2x"] = addFile(`${stem}${scale === 2 ? "@2x" : ""}.png`, image.asPng(), { width: image.width, height: image.height });
            }
            manifest.images.push(entry);
        }
        console.log(`Exported ${format.id}: ${layout.nodes.length} papers, ${layout.labels.length} topics`);
    }
    addFile("manifest.json", JSON.stringify(manifest, null, 2) + "\n");
    addFile("index.html", gallery(manifest));
    addFile("README.txt", `CHRIS MCCOMB · PUBLICATION MAPS\nSite v${manifest.version} · ${manifest.updated}\n\n${manifest.publications} publications in ${manifest.topics} research topics.\n\nFORMATS\n${FORMATS.map(f => `${f.name}: ${f.width} x ${f.height} PNG; ${f.width * 2} x ${f.height * 2} PNG @2x; SVG`).join("\n")}\n\nEach format includes dark and light themes. Use the native PNG for social uploads,\nthe @2x PNG for high-resolution slides, or the SVG for unrestricted scaling.\nSVG text is outlined, so no font installation or external resource is needed.\nKeep the original aspect ratio when inserting images.\n\nThe LinkedIn banner reserves the lower-left 23% of width and 44% of height\nfor a profile photo. Platform crops and overlays vary by device; inspect the\nupload preview. No banner layout can guarantee every platform crop.\n\nREADING THE MAP\nEach circle is one publication. Color uses the website's Magma palette for year;\ncircle radius is a square-root scale of citations, including a minimum visible\nsize for uncited papers. The legend uses the same sizes as the map. Positions\ncome from the site's topic embedding, fitted to each aspect ratio and adjusted\nto avoid collisions, labels, and legends. Absolute distances are not a metric.\nTopics use the same display labels as the interactive map.\n\nDATA AND REPRODUCIBILITY\nSnapshot built: ${manifest.dataBuiltAt}\nDataset commit: ${manifest.datasetCommit}\nSnapshot SHA-256: ${manifest.snapshotSha256}\nLatest source: ${manifest.source}\nExporting does not refresh papers or citation counts.\nThe manifest records every image's dimensions and SHA-256.\nRun npm ci and npm run export:maps from the site repository to regenerate.\n\nLICENSE\nMap exports follow the repository's MIT license (LICENSE.txt).\nInter typography by the Inter Project Authors, SIL Open Font License 1.1\n(font-license.txt). SVGs contain outlined glyphs, not the font file.\nD3 7.9.0 uses the ISC license (d3-license.txt).\n`);
    addFile("LICENSE.txt", fs.readFileSync(path.join(ROOT, "LICENSE.md")));
    addFile("font-license.txt", fs.readFileSync(path.join(__dirname, "fonts/OFL.txt")));
    addFile("d3-license.txt", fs.readFileSync(path.join(ROOT, "assets/vendor/d3/LICENSE")));
    const archive = zipSync(Object.fromEntries(Object.entries(files).map(([name, bytes]) =>
        [name, [bytes, { mtime: new Date(`${release.last_updated}T12:00:00`) }]])), { level: 6 });
    // Generate and validate everything before writing outputs; never remove an output directory.
    fs.mkdirSync(outputDir, { recursive: true });
    for (const [name, bytes] of Object.entries(files)) fs.writeFileSync(path.join(outputDir, name), bytes);
    fs.writeFileSync(path.join(outputDir, "publication-maps.zip"), archive);
    return manifest;
}

if (require.main === module) {
    const { values } = parseArgs({ options: { output: { type: "string" } } });
    const output = values.output ? path.resolve(values.output) : path.join(ROOT, "_map_exports");
    buildExports(output);
    console.log(`Map gallery and ZIP: ${output}`);
}
module.exports = { buildExports };
