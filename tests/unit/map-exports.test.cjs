"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { createHash } = require("node:crypto");
const { unzipSync } = require("fflate");
const { FORMATS, layoutMap, validateData, validateLayout, mapSVG } = require("../../_scripts/map_exports.cjs");
const { buildExports } = require("../../_scripts/export_publication_maps.cjs");
const data = require("../../assets/json/pubs.json");

test("every aspect ratio preserves papers and topics without clipping or collisions", () => {
    for (const format of FORMATS) {
        const layout = layoutMap(data, format);
        assert.deepEqual(new Set(layout.nodes.map(n => n.id)), new Set(data.records.map(r => r.author_pub_id)));
        assert.equal(layout.labels.length, data.clusters.length);
        assert(layout.labels.some(l => l.text === "design teams"));
        assert.doesNotThrow(() => validateLayout(layout));
        assert.equal(layout.nodes.find(n => n.year === layout.years[0]).color, "#000004");
        assert.equal(layout.nodes.find(n => n.year === layout.years[1]).color, "#fcfdbf");
        if (format.banner) assert(layout.reserved.some(r => r.x === 0 && r.width === format.width * .23));
    }
});

test("layouts and self-contained SVGs are reproducible and retain accessible descriptions", () => {
    const a = layoutMap(data, FORMATS[0]);
    const b = layoutMap(data, FORMATS[0]);
    assert.deepEqual(a, b);
    const svg = mapSVG(a, "dark");
    assert.equal(svg, mapSVG(b, "dark"));
    assert.match(svg, /<title>Chris McComb/);
    assert(svg.includes(`<desc>${data.records.length} publications`));
    assert.doesNotMatch(svg, /<text\b|<image\b|<script\b|@import|@font-face/);
    assert.match(svg, /<path/);
});

test("invalid source data and obscured or clipped papers fail the build", () => {
    const bad = structuredClone(data);
    bad.records.push(bad.records[0]);
    assert.throws(() => validateData(bad), /duplicate/);
    bad.records = [{ ...data.records[0], x: null }];
    assert.throws(() => validateData(bad), /Invalid/);
    const layout = layoutMap(data, FORMATS[0]);
    layout.nodes[0].x = -1;
    assert.throws(() => validateLayout(layout), /Clipped/);
    layout.nodes[0].x = layout.labels[0].x + layout.labels[0].width / 2;
    layout.nodes[0].y = layout.labels[0].y + layout.labels[0].height / 2;
    assert.throws(() => validateLayout(layout), /Obscured/);
});

test("export bundle contains every requested resolution, matching checksums, and working gallery links", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "map-export-test-"));
    try {
        const manifest = buildExports(directory);
        assert.equal(manifest.images.length, 12);
        assert.equal(manifest.publications, data.records.length);
        const archive = unzipSync(fs.readFileSync(path.join(directory, "publication-maps.zip")));
        const html = fs.readFileSync(path.join(directory, "index.html"), "utf8");
        for (const image of manifest.images) for (const [kind, file] of Object.entries(image.files)) {
            const buffer = fs.readFileSync(path.join(directory, file.path));
            assert.equal(file.sha256, createHash("sha256").update(buffer).digest("hex"));
            assert.equal(buffer.length, file.bytes);
            assert.deepEqual(Buffer.from(archive[file.path]), buffer);
            assert(html.includes(`href="${file.path}"`));
            if (kind.startsWith("png")) {
                assert.equal(buffer.subarray(1, 4).toString(), "PNG");
                assert.equal(buffer.readUInt32BE(16), file.width);
                assert.equal(buffer.readUInt32BE(20), file.height);
                assert.equal(file.width, image.width * (kind === "png2x" ? 2 : 1));
                assert.equal(file.height, image.height * (kind === "png2x" ? 2 : 1));
            }
        }
        assert.equal(Object.keys(archive).filter(n => n.endsWith(".png")).length, 24);
        assert.equal(Object.keys(archive).filter(n => n.endsWith(".svg")).length, 12);
        assert(archive["README.txt"] && archive["font-license.txt"] && archive["manifest.json"]);
    } finally { fs.rmSync(directory, { recursive: true, force: true }); }
});
