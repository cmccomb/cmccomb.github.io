/* Deterministic figure layout, sharing the site's snapshot, palette, and labels. */
"use strict";

const path = require("node:path");
const d3 = require("../assets/vendor/d3/d3.v7.9.0.min.js");
const { Resvg } = require("@resvg/resvg-js");
const { displayClusterLabel } = require("../assets/js/publication_helpers.js");

const FORMATS = [
    { id: "linkedin-banner", name: "LinkedIn profile banner", width: 1584, height: 396, font: 18, radius: 7.5, banner: true },
    { id: "social-landscape", name: "Social landscape", width: 1200, height: 627, font: 20, radius: 10 },
    { id: "social-square", name: "Social square", width: 1080, height: 1080, font: 24, radius: 13 },
    { id: "social-portrait", name: "Social portrait", width: 1080, height: 1350, font: 25, radius: 14 },
    { id: "slides-16x9", name: "Widescreen slides · 16:9", width: 1920, height: 1080, font: 30, radius: 16 },
    { id: "slides-4x3", name: "Standard slides · 4:3", width: 1600, height: 1200, font: 30, radius: 16 },
];
const THEMES = {
    dark: { background: "#212529", text: "#eef0f2", muted: "#b9bfc5", border: "#dce1e6", label: "#f7f8f9", labelText: "#212529" },
    light: { background: "#ffffff", text: "#212529", muted: "#58616a", border: "#545b62", label: "#f3f4f5", labelText: "#212529" },
};
const FONT_OPTIONS = {
    fontFiles: [path.join(__dirname, "fonts/Inter.ttf")],
    loadSystemFonts: false,
    defaultFontFamily: "Inter",
};
const xml = value => String(value).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[c]));
const clamp = (n, min, max) => Math.max(min, Math.min(max, n));
const round = n => +n.toFixed(3);

function textWidth(text, fontSize) {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="2000" height="100"><text x="0" y="60" font-family="Inter" font-size="${fontSize}" font-weight="600">${xml(text)}</text></svg>`;
    return new Resvg(svg, { font: FONT_OPTIONS }).getBBox().width;
}

function rectanglesOverlap(a, b, gap = 0) {
    return a.x < b.x + b.width + gap && a.x + a.width + gap > b.x
        && a.y < b.y + b.height + gap && a.y + a.height + gap > b.y;
}

function circleHitsRect(node, rect, gap = 0) {
    const dx = node.x - clamp(node.x, rect.x, rect.x + rect.width);
    const dy = node.y - clamp(node.y, rect.y, rect.y + rect.height);
    return dx * dx + dy * dy < (node.r + gap) ** 2;
}

// Move circles out through the closest side; keep a gap around labels and legends.
function excludeObstacles(node, obstacles, gap, bounds) {
    if (!obstacles.some(rect => circleHitsRect(node, rect, gap))) return;
    const choices = obstacles.flatMap(rect => {
        const left = rect.x - node.r - gap - .01;
        const right = rect.x + rect.width + node.r + gap + .01;
        const top = rect.y - node.r - gap - .01;
        const bottom = rect.y + rect.height + node.r + gap + .01;
        return [
            { x: left, y: node.y }, { x: right, y: node.y },
            { x: node.x, y: top }, { x: node.x, y: bottom },
            { x: left, y: top }, { x: right, y: top },
            { x: left, y: bottom }, { x: right, y: bottom },
        ];
    }).filter(p => p.x >= bounds.margin + node.r && p.x <= bounds.width - bounds.margin - node.r
        && p.y >= bounds.margin + node.r && p.y <= bounds.height - bounds.margin - node.r
        && !obstacles.some(r => circleHitsRect({ ...p, r: node.r }, r, gap)))
        .sort((a, b) => Math.hypot(a.x - node.x, a.y - node.y) - Math.hypot(b.x - node.x, b.y - node.y));
    if (!choices.length) throw new Error("No room outside obstacle");
    node.x = choices[0].x;
    node.y = choices[0].y;
    node.vx = 0;
    node.vy = 0;
}

function validateData(data) {
    if (!Array.isArray(data.records) || !data.records.length || !Array.isArray(data.clusters)) {
        throw new Error("A publication snapshot with records and clusters is required");
    }
    const clusters = new Set(data.clusters.map(c => String(c.id)));
    const ids = new Set();
    for (const record of data.records) {
        if (!record.author_pub_id || ids.has(record.author_pub_id)
            || ![record.x, record.y, record.pub_year, record.num_citations].every(Number.isFinite)
            || record.num_citations < 0 || !clusters.has(String(record.cluster_id))) {
            throw new Error("Invalid or duplicate publication, or missing topic");
        }
        ids.add(record.author_pub_id);
    }
    if (clusters.size !== data.clusters.length || data.clusters.some(c => !displayClusterLabel(c.label))) {
        throw new Error("Invalid or duplicate topic");
    }
}

function layoutMap(data, format) {
    validateData(data);
    const { width, height, font, radius, banner } = format;
    const margin = banner ? 20 : Math.round(width * .025);
    const legend = { x: width - margin - font * 21, y: margin, width: font * 21, height: font * 4.5 };
    const reserved = [legend];
    if (banner) reserved.push({ x: 0, y: height * .56, width: width * .23, height: height * .44 });
    const x = d3.scaleLinear().domain(d3.extent(data.records, d => d.x)).range([margin + radius * 2, width - margin - radius * 2]);
    const y = d3.scaleLinear().domain(d3.extent(data.records, d => d.y)).range([height - margin - radius * 2, margin + radius * 2]);
    const years = d3.extent(data.records, d => d.pub_year);
    const citations = d3.extent(data.records, d => d.num_citations);
    const color = d3.scaleSequential(d3.interpolateMagma).domain(years);
    const size = d3.scaleSqrt().domain(citations).range([radius * .7, radius * 1.45]);
    const nodes = data.records.map(d => ({
        id: d.author_pub_id, cluster: String(d.cluster_id), year: d.pub_year, citations: d.num_citations,
        x: x(d.x), y: y(d.y), targetX: x(d.x), targetY: y(d.y), r: size(d.num_citations), color: color(d.pub_year),
    }));
    function constrain(obstacles) {
        for (const n of nodes) {
            n.x = clamp(n.x, margin + n.r, width - margin - n.r);
            n.y = clamp(n.y, margin + n.r, height - margin - n.r);
            excludeObstacles(n, obstacles, 4, { width, height, margin });
        }
    }
    function simulate(obstacles) {
        const simulation = d3.forceSimulation(nodes)
            .randomSource(d3.randomLcg(42))
            .force("x", d3.forceX(d => d.targetX).strength(.09))
            .force("y", d3.forceY(d => d.targetY).strength(.09))
            .force("collide", d3.forceCollide(d => d.r + 1.5).iterations(4))
            .stop();
        for (let i = 0; i < 360; i++) { simulation.tick(); constrain(obstacles); }
    }
    simulate(reserved);
    const groups = d3.group(nodes, d => d.cluster);
    const topics = data.clusters.map(c => ({
        id: String(c.id), text: displayClusterLabel(c.label), count: groups.get(String(c.id))?.length || 0,
        cx: d3.mean(groups.get(String(c.id)) || [], d => d.x),
        cy: d3.mean(groups.get(String(c.id)) || [], d => d.y),
    })).filter(c => c.count).sort((a, b) => b.count - a.count || a.id.localeCompare(b.id));
    const labels = [];
    for (const topic of topics) {
        const w = Math.ceil(textWidth(topic.text, font) + font * .85);
        const h = Math.ceil(font * 1.6);
        const candidates = [];
        for (let ring = 0; ring <= 15; ring++) {
            const steps = ring ? 24 : 1;
            for (let angle = 0; angle < steps; angle++) {
                const a = angle * 2 * Math.PI / steps;
                const candidate = {
                    ...topic, width: w, height: h,
                    x: clamp(topic.cx + Math.cos(a) * ring * font * .85 - w / 2, margin, width - margin - w),
                    y: clamp(topic.cy + Math.sin(a) * ring * font * .85 - h / 2, margin, height - margin - h),
                };
                if ([...reserved, ...labels].some(r => rectanglesOverlap(candidate, r, font * .5))) continue;
                candidate.score = Math.hypot(candidate.x + w / 2 - topic.cx, candidate.y + h / 2 - topic.cy)
                    + nodes.filter(n => circleHitsRect(n, candidate, 4)).length * font * .32;
                candidates.push(candidate);
            }
        }
        candidates.sort((a, b) => a.score - b.score);
        if (!candidates.length) throw new Error(`No room for topic ${topic.text} in ${format.id}`);
        labels.push(candidates[0]);
    }
    simulate([...reserved, ...labels]);
    // Resolve residual circle contacts after constrained force ticks, without drift.
    for (let iteration = 0; iteration < 600; iteration++) {
        let overlap = 0;
        for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
            const a = nodes[i], b = nodes[j];
            const dx = b.x - a.x, dy = b.y - a.y;
            const distance = Math.hypot(dx, dy);
            const penetration = a.r + b.r + 1 - distance;
            if (penetration <= 0) continue;
            overlap = Math.max(overlap, penetration);
            const ux = distance ? dx / distance : 1;
            const uy = distance ? dy / distance : 0;
            a.x -= ux * penetration / 2; a.y -= uy * penetration / 2;
            b.x += ux * penetration / 2; b.y += uy * penetration / 2;
        }
        constrain([...reserved, ...labels]);
        if (overlap < .05) break;
    }
    const layout = { format, margin, legend, reserved, nodes, labels, years, citations };
    validateLayout(layout);
    return layout;
}

function validateLayout({ format, nodes, labels, reserved }) {
    for (const n of nodes) {
        if (n.x - n.r < 0 || n.x + n.r > format.width || n.y - n.r < 0 || n.y + n.r > format.height) {
            throw new Error(`Clipped paper in ${format.id}`);
        }
        if ([...labels, ...reserved].some(r => circleHitsRect(n, r))) throw new Error(`Obscured paper ${n.id} in ${format.id}`);
    }
    for (const [i, a] of nodes.entries()) for (const b of nodes.slice(i + 1)) {
        if (Math.hypot(a.x - b.x, a.y - b.y) < a.r + b.r - .1) throw new Error(`Overlapping papers in ${format.id}`);
    }
    for (const [i, label] of labels.entries()) {
        if (label.x < 0 || label.y < 0 || label.x + label.width > format.width || label.y + label.height > format.height
            || [...labels.slice(i + 1), ...reserved].some(r => rectanglesOverlap(label, r))) {
            throw new Error(`Clipped or overlapping topic in ${format.id}`);
        }
    }
}

function mapSVG(layout, themeName) {
    const theme = THEMES[themeName];
    if (!theme) throw new Error(`Unknown theme: ${themeName}`);
    const { format: { width, height, font, radius, name }, nodes, labels, years, citations, legend } = layout;
    const lFont = font * .67;
    const barX = legend.x + font * .4, barY = legend.y + font * 1.4;
    const barW = font * 10, barH = font * .43;
    const sizeX = barX + font * 12;
    const size = d3.scaleSqrt().domain(citations).range([radius * .7, radius * 1.45]);
    const largestRadius = size(citations[1]);
    const sizes = [...new Set([citations[0], Math.round(citations[1] / 4), citations[1]])];
    const description = `${nodes.length} publications, ${labels.length} research topics. Color represents publication year (${years.join("–")}); circle radius scales with the square root of citations. Positions adapt the website's embedding to ${name.toLowerCase()}.`;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title description">
<title id="title">Chris McComb · publication map</title><desc id="description">${xml(description)}</desc>
<defs><linearGradient id="years">${d3.range(41).map(i => `<stop offset="${i / 40}" stop-color="${d3.interpolateMagma(i / 40)}"/>`).join("")}</linearGradient></defs>
<rect width="${width}" height="${height}" fill="${theme.background}"/>
<g stroke="${theme.border}" stroke-width="${round(font * .026)}" stroke-opacity=".45">${nodes.map(n => `<circle cx="${round(n.x)}" cy="${round(n.y)}" r="${round(n.r)}" fill="${n.color}"/>`).join("")}</g>
<g font-family="Inter" font-size="${font}" font-weight="600">${labels.map(l => `<rect x="${round(l.x)}" y="${round(l.y)}" width="${l.width}" height="${l.height}" rx="${round(font * .27)}" fill="${theme.label}"/><text x="${round(l.x + l.width / 2)}" y="${round(l.y + l.height / 2 + font * .35)}" text-anchor="middle" fill="${theme.labelText}">${xml(l.text)}</text>`).join("")}</g>
<g font-family="Inter" font-size="${lFont}" fill="${theme.text}">
<text x="${barX}" y="${legend.y + lFont}" font-weight="600">Publication year</text>
<rect x="${barX}" y="${barY}" width="${barW}" height="${barH}" rx="${barH / 2}" fill="url(#years)" stroke="${theme.muted}" stroke-width=".5"/>
${[years[0], Math.round((years[0] + years[1]) / 2), years[1]].map((year, i) => `<text x="${barX + i * barW / 2}" y="${barY + font * 1.23}" text-anchor="${i === 0 ? "start" : i === 2 ? "end" : "middle"}" fill="${theme.muted}">${year}</text>`).join("")}
<text x="${sizeX}" y="${legend.y + lFont}" font-weight="600">Citations</text>
${sizes.map((n, i) => `<circle cx="${sizeX + font * .8 + i * font * 2.7}" cy="${barY + largestRadius * 2 - size(n)}" r="${size(n)}" fill="none" stroke="${theme.muted}" stroke-width="1"/><text x="${sizeX + font * .8 + i * font * 2.7}" y="${barY + largestRadius * 2 + lFont * 1.6}" text-anchor="middle" fill="${theme.muted}">${n}</text>`).join("")}
</g></svg>`;
    // Outlined text keeps SVGs portable in slide software with no font install.
    const outlined = new Resvg(svg, { font: FONT_OPTIONS }).toString();
    return outlined.replace(/(<svg\b[^>]*>)/, `$1\n<title>Chris McComb · publication map</title><desc>${xml(description)}</desc>`);
}

module.exports = { FORMATS, THEMES, FONT_OPTIONS, xml, validateData, layoutMap, validateLayout, mapSVG };
