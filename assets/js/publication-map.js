const MIN_PUBLICATION_RADIUS = 10.5;
// Keep touch targets above 24 px in the full-size map.
const MIN_PUBLICATION_HIT_RADIUS = 12.1;
const MIN_VISIBLE_TOPIC_SIZE = 3;

/**
 * Draw the map and its legends. The browser controller owns search, selection,
 * keyboard behavior, and URL state. D3 updates the supplied nodes in place.
 */
export function createPublicationMap({ svg, graphContainer, mapViewport, nodes, colorScale, hideTooltip }) {
    const { d3 } = window;
    const fitMap = graphContainer.querySelector("#publication-fit-map");
    const legendContainer = d3.select(graphContainer.querySelector(".colorbar-legend"));
    const sizeLegendContainer = d3.select(graphContainer.querySelector(".size-legend"));
    const xScale = d3.scaleLinear().domain(d3.extent(nodes, node => node.x_data));
    const yScale = d3.scaleLinear().domain(d3.extent(nodes, node => node.y_data));
    const yearExtent = colorScale.domain();
    const radiusExtent = d3.extent(nodes, node => node.num_citations);
    const radiusScale = d3.scaleSqrt().domain(radiusExtent);

    for (const node of nodes) {
        // Keep the force simulation's starting positions deterministic.
        node.x = 0;
        node.y = 0;
    }

    const legendWidth = 150;
    const legendHeight = 10;
    const legendMargins = { top: 18, right: 8, bottom: 24, left: 8 };
    const gradientId = "publication-year-gradient";
    const legendSvg = legendContainer.append("svg")
        .attr("class", "colorbar-svg")
        .attr("width", legendWidth + legendMargins.left + legendMargins.right)
        .attr("height", legendHeight + legendMargins.top + legendMargins.bottom)
        .attr(
            "viewBox",
            `0 0 ${legendWidth + legendMargins.left + legendMargins.right} ${legendHeight + legendMargins.top + legendMargins.bottom}`
        );
    const gradient = legendSvg.append("defs")
        .append("linearGradient")
        .attr("id", gradientId)
        .attr("x1", "0%")
        .attr("x2", "100%")
        .attr("y1", "0%")
        .attr("y2", "0%");

    gradient.selectAll("stop")
        .data(d3.range(0, 1.0001, 0.05))
        .enter()
        .append("stop")
        .attr("offset", value => `${value * 100}%`)
        .attr("stop-color", value => colorScale(
            yearExtent[0] + value * (yearExtent[1] - yearExtent[0])
        ));

    legendSvg.append("rect")
        .attr("x", legendMargins.left)
        .attr("y", legendMargins.top)
        .attr("width", legendWidth)
        .attr("height", legendHeight)
        .attr("rx", 6)
        .attr("fill", `url(#${gradientId})`);

    const legendScale = d3.scaleLinear().domain(yearExtent).range([0, legendWidth]);
    const legendTicks = yearExtent[0] === yearExtent[1]
        ? [yearExtent[0]]
        : Array.from(
            new Set(d3.range(4).map(index => Math.round(
                yearExtent[0] + index * (yearExtent[1] - yearExtent[0]) / 3
            )))
        );
    const legendAxis = d3.axisBottom(legendScale)
        .tickValues(legendTicks)
        .tickFormat(d3.format("d"));
    const axisGroup = legendSvg.append("g")
        .attr("transform", `translate(${legendMargins.left}, ${legendMargins.top + legendHeight})`)
        .call(legendAxis);

    axisGroup.selectAll("text")
        .attr("fill", "#f8f9fa")
        .attr("font-size", 10);
    axisGroup.selectAll(".tick:first-of-type text").attr("text-anchor", "start");
    axisGroup.selectAll(".tick:last-of-type text").attr("text-anchor", "end");
    axisGroup.selectAll("line, path")
        .attr("stroke", "rgba(248, 249, 250, 0.4)");
    legendSvg.append("text")
        .attr("x", legendMargins.left)
        .attr("y", legendMargins.top - 6)
        .attr("fill", "#f8f9fa")
        .attr("font-size", 12)
        .attr("font-weight", 600)
        .text("Year");

    const sizeLegendWidth = 140;
    const sizeLegendHeight = 76;
    const sizeLegendSvg = sizeLegendContainer.append("svg")
        .attr("class", "size-legend-svg")
        .attr("width", sizeLegendWidth)
        .attr("height", sizeLegendHeight)
        .attr("viewBox", `0 0 ${sizeLegendWidth} ${sizeLegendHeight}`);
    sizeLegendSvg.append("text")
        .attr("x", 8)
        .attr("y", 10)
        .attr("fill", "#f8f9fa")
        .attr("font-size", 12)
        .attr("font-weight", 600)
        .text("Citations");

    function renderSizeLegend() {
        const values = radiusExtent[0] === radiusExtent[1]
            ? [radiusExtent[0]]
            : [
                radiusExtent[0],
                radiusExtent[0] + (radiusExtent[1] - radiusExtent[0]) / 4,
                radiusExtent[1],
            ];
        const radii = values.map(value => radiusScale(value));
        const firstCenter = 20;
        const lastCenter = 116;
        const positions = values.length === 1
            ? [sizeLegendWidth / 2]
            : [
                firstCenter,
                // Match the gaps between circle edges as the radii change.
                (firstCenter + radii[0] + lastCenter - radii[2]) / 2,
                lastCenter,
            ];
        const legendEntries = values.map((value, index) => ({
            value,
            x: positions[index],
            radius: radii[index],
        }));
        const sizeKeys = sizeLegendSvg.selectAll("g.citation-size-key")
            .data(legendEntries, entry => entry.value)
            .join(enter => {
                const group = enter.append("g").attr("class", "citation-size-key");
                group.append("circle")
                    .attr("fill", "rgba(248, 249, 250, 0.18)")
                    .attr("stroke", "#f8f9fa")
                    .attr("stroke-width", 1.25);
                group.append("text")
                    .attr("fill", "#f8f9fa")
                    .attr("font-size", 10)
                    .attr("text-anchor", "middle");
                return group;
            });
        sizeKeys.attr("transform", entry => `translate(${entry.x}, 35)`);
        sizeKeys.select("circle").attr("r", entry => entry.radius);
        sizeKeys.select("text")
            .attr("y", 37)
            .text(entry => d3.format(",d")(Math.round(entry.value)));
        sizeLegendContainer.attr(
            "aria-label",
            `Circle sizes represent citation counts from ${Math.round(radiusExtent[0])} to ${Math.round(radiusExtent[1])}.`
        );
    }

    const publicationControls = svg.selectAll("g.publication-link")
        .data(nodes)
        .enter()
        .append("g")
        .attr("class", "publication-link")
        .attr("role", "button")
        .attr("aria-controls", "publication-detail")
        .attr("aria-expanded", "false")
        .attr("tabindex", -1)
        .attr("aria-describedby", "publication-tooltip")
        .attr("aria-label", node => (
            `${node.title}; ${node.pub_year}; ${node.num_citations} citation${node.num_citations === 1 ? "" : "s"}; map neighborhood ${node.cluster_label}; show publication details`
        ));
    const hitTargetSelection = publicationControls.append("rect")
        .attr("class", "publication-hit-target")
        .attr("fill", "transparent")
        .attr("aria-hidden", "true");
    const nodeSelection = publicationControls.append("rect")
        .attr("class", "publication-node")
        .attr("fill", node => node.color)
        .attr("opacity", 1);

    const clusterLabelLayer = svg.append("g")
        .attr("class", "cluster-label-layer")
        .attr("aria-hidden", "true");
    let hideClusterLabel = () => false;
    let simulation;
    let renderedDimensions = "";
    let mapScroll = { left: 0, top: 0 };

    fitMap.addEventListener("click", () => {
        const fitted = !mapViewport.classList.contains("map-fitted");
        if (fitted) mapScroll = { left: mapViewport.scrollLeft, top: mapViewport.scrollTop };
        mapViewport.classList.toggle("map-fitted", fitted);
        fitMap.setAttribute("aria-pressed", String(fitted));
        mapViewport.scrollTo(fitted ? { left: 0, top: 0 } : mapScroll);
        hideTooltip();
    });

    function render() {
        hideTooltip();
        if (!mapViewport.clientWidth || !mapViewport.clientHeight) return;
        // Preserve readable labels and touch targets instead of squeezing the
        // entire collection into a phone-sized plot.
        const width = Math.max(1024, mapViewport.clientWidth);
        const height = Math.max(720, mapViewport.clientHeight);
        const fitScale = Math.min(mapViewport.clientWidth / width, mapViewport.clientHeight / height);
        fitMap.hidden = fitScale === 1;
        mapViewport.classList.toggle("small-map-fit", fitScale < 0.65);
        const dimensions = `${width},${height},${window.innerWidth}`;
        if (dimensions === renderedDimensions) return;
        renderedDimensions = dimensions;
        svg.attr("width", width)
            .attr("height", height)
            .attr("viewBox", `0 0 ${width} ${height}`);

        const mapPadding = 40;
        const topPadding = mapPadding;
        const bottomCoordinate = Math.max(topPadding + 1, height - mapPadding);
        xScale.range([mapPadding, width - mapPadding]);
        yScale.range([bottomCoordinate, topPadding]);
        const radiusBase = Math.sqrt(width * height);
        const maxRadius = Math.min(20, Math.max(16, radiusBase / 60)) * 1.1;
        radiusScale.range([MIN_PUBLICATION_RADIUS, maxRadius]);
        renderSizeLegend();
        nodes.forEach(node => {
            node.x_orig = xScale(node.x_data);
            node.y_orig = yScale(node.y_data);
            node.r = radiusScale(node.num_citations);
            node.hitRadius = Math.max(MIN_PUBLICATION_HIT_RADIUS, node.r);
        });

        if (!simulation) {
            simulation = d3.forceSimulation(nodes)
                .force("x", d3.forceX(node => node.x_orig).strength(0.15))
                .force("y", d3.forceY(node => node.y_orig).strength(0.15))
                .force("collide", d3.forceCollide(node => node.hitRadius + 1).strength(1).iterations(3))
                .stop();
        } else {
            simulation
                .force("x", d3.forceX(node => node.x_orig).strength(0.15))
                .force("y", d3.forceY(node => node.y_orig).strength(0.15))
                .force("collide", d3.forceCollide(node => node.hitRadius + 1).strength(1).iterations(3));
        }

        simulation.alpha(1).stop();
        const constrainNodeToMap = node => {
            const minimumX = node.hitRadius;
            const maximumX = width - node.hitRadius;
            const minimumY = topPadding + node.hitRadius;
            const maximumY = bottomCoordinate - node.hitRadius;
            node.x = minimumX <= maximumX
                ? Math.max(minimumX, Math.min(maximumX, node.x))
                : width / 2;
            node.y = minimumY <= maximumY
                ? Math.max(minimumY, Math.min(maximumY, node.y))
                : (topPadding + bottomCoordinate) / 2;
        };
        for (let index = 0; index < 200; index += 1) {
            simulation.tick();
            nodes.forEach(constrainNodeToMap);
        }
        const clusteredNodes = nodes.filter(node => (
            node.cluster_id !== null
            && node.cluster_id !== undefined
            && Number.isFinite(node.x)
            && Number.isFinite(node.y)
        ));
        if (clusteredNodes.length === 0) {
            clusterLabelLayer.selectAll("g.cluster-label").remove();
            return;
        }

        const labelData = Array.from(
            d3.group(clusteredNodes, node => node.cluster_id),
            ([clusterId, members]) => {
                return {
                    clusterId,
                    count: members.length,
                    label: members[0].cluster_label,
                    x: d3.mean(members, member => member.x) ?? 0,
                    y: d3.mean(members, member => member.y) ?? 0,
                };
            }
        ).filter(label => label.count >= MIN_VISIBLE_TOPIC_SIZE)
            .sort((a, b) => b.count - a.count || String(a.clusterId).localeCompare(String(b.clusterId)));
        const mergedLabels = clusterLabelLayer.selectAll("g.cluster-label")
            .data(labelData, label => label.clusterId)
            .join(enter => {
                const group = enter.append("g").attr("class", "cluster-label");
                group.append("rect").attr("class", "cluster-label-background");
                group.append("text")
                    .attr("class", "cluster-label-text")
                    .attr("text-anchor", "middle")
                    .attr("dominant-baseline", "middle");
                return group;
            });
        mergedLabels.style("display", null).select("text").text(label => label.label);
        const placedLabels = [];
        const circleHitsLabel = (node, rect, gap = 4) => {
            const dx = node.x - Math.max(rect.x, Math.min(node.x, rect.x + rect.width));
            const dy = node.y - Math.max(rect.y, Math.min(node.y, rect.y + rect.height));
            return dx * dx + dy * dy < (node.hitRadius + gap) ** 2;
        };
        const labelsOverlap = (a, b) => (
            a.x < b.x + b.width + 8 && a.x + a.width + 8 > b.x
            && a.y < b.y + b.height + 8 && a.y + a.height + 8 > b.y
        );
        mergedLabels.each(function sizeAndPositionLabel(label) {
            const group = d3.select(this);
            const boundingBox = group.select("text").node().getBBox();
            const localX = boundingBox.x - 6;
            const localY = boundingBox.y - 4;
            const labelWidth = boundingBox.width + 12;
            const labelHeight = boundingBox.height + 8;
            const margin = 8;
            if (labelWidth > width - 2 * margin || labelHeight > height - 2 * margin) {
                throw new Error("Publication topic label exceeds the map bounds");
            }
            const candidates = [];
            const step = 14;
            // Stay near the topic, preferring gaps that need fewer bubbles
            // moved. Larger rings are only needed if other labels fill the area.
            for (let ring = 0; ring <= Math.ceil(Math.hypot(width, height) / step); ring += 1) {
                if (ring > 12 && candidates.length) break;
                const angles = ring ? 24 : 1;
                for (let angle = 0; angle < angles; angle += 1) {
                    const radians = angle * 2 * Math.PI / angles;
                    const rect = {
                        x: Math.max(margin, Math.min(width - margin - labelWidth,
                            label.x + Math.cos(radians) * ring * step - labelWidth / 2)),
                        y: Math.max(margin, Math.min(height - margin - labelHeight,
                            label.y + Math.sin(radians) * ring * step - labelHeight / 2)),
                        width: labelWidth,
                        height: labelHeight,
                    };
                    if (placedLabels.some(other => labelsOverlap(rect, other))) continue;
                    rect.score = Math.hypot(rect.x + labelWidth / 2 - label.x,
                        rect.y + labelHeight / 2 - label.y)
                        + nodes.filter(node => circleHitsLabel(node, rect)).length * labelHeight * .5;
                    candidates.push(rect);
                }
            }
            candidates.sort((a, b) => a.score - b.score || a.x - b.x || a.y - b.y);
            const position = candidates[0];
            if (!position) throw new Error("No room for publication topic labels");
            placedLabels.push(position);
            // Use the actual text bounds rather than assuming its baseline
            // centers the padded rectangle around the group's origin.
            group.attr("transform", `translate(${position.x - localX}, ${position.y - localY})`)
                .select("rect")
                .attr("x", localX)
                .attr("y", localY)
                .attr("width", labelWidth)
                .attr("height", labelHeight)
                .attr("rx", 8)
                .attr("ry", 8);
        });

        const keepClearOfLabels = node => {
            constrainNodeToMap(node);
            if (!placedLabels.some(rect => circleHitsLabel(node, rect))) return;
            const gap = 4.01;
            const choices = placedLabels.flatMap(rect => {
                const left = rect.x - node.hitRadius - gap;
                const right = rect.x + rect.width + node.hitRadius + gap;
                const top = rect.y - node.hitRadius - gap;
                const bottom = rect.y + rect.height + node.hitRadius + gap;
                return [
                    { x: left, y: node.y }, { x: right, y: node.y },
                    { x: node.x, y: top }, { x: node.x, y: bottom },
                    { x: left, y: top }, { x: right, y: top },
                    { x: left, y: bottom }, { x: right, y: bottom },
                ];
            }).filter(point => (
                point.x >= node.hitRadius && point.x <= width - node.hitRadius
                && point.y >= topPadding + node.hitRadius && point.y <= bottomCoordinate - node.hitRadius
                && !placedLabels.some(rect => circleHitsLabel({ ...point, hitRadius: node.hitRadius }, rect))
            ));
            choices.sort((a, b) => Math.hypot(a.x - node.x, a.y - node.y)
                - Math.hypot(b.x - node.x, b.y - node.y));
            if (!choices.length) throw new Error("No clear position for publication bubble");
            node.x = choices[0].x;
            node.y = choices[0].y;
            node.vx = 0;
            node.vy = 0;
        };
        // Labels are fixed obstacles in the second pass. This reserves a
        // small gap around every label while retaining the projection forces.
        simulation.alpha(.45).stop();
        for (let index = 0; index < 180; index += 1) {
            simulation.tick();
            nodes.forEach(keepClearOfLabels);
        }
        // Resolve contacts introduced by clamping a bubble out of a label.
        for (let iteration = 0; iteration < 600; iteration += 1) {
            let overlap = 0;
            for (let i = 0; i < nodes.length; i += 1) {
                for (let j = i + 1; j < nodes.length; j += 1) {
                    const a = nodes[i];
                    const b = nodes[j];
                    const dx = b.x - a.x;
                    const dy = b.y - a.y;
                    const distance = Math.hypot(dx, dy);
                    const penetration = a.hitRadius + b.hitRadius + 1 - distance;
                    if (penetration <= 0) continue;
                    overlap = Math.max(overlap, penetration);
                    const ux = distance ? dx / distance : 1;
                    const uy = distance ? dy / distance : 0;
                    a.x -= ux * penetration / 2;
                    a.y -= uy * penetration / 2;
                    b.x += ux * penetration / 2;
                    b.y += uy * penetration / 2;
                }
            }
            nodes.forEach(keepClearOfLabels);
            if (overlap < .05) break;
        }
        if (nodes.some((node, index) => (
            placedLabels.some(rect => circleHitsLabel(node, rect, 0))
            || nodes.slice(index + 1).some(other => (
                Math.hypot(node.x - other.x, node.y - other.y) < node.hitRadius + other.hitRadius - .1
            ))
        ))) {
            throw new Error("Could not settle publication map without overlaps");
        }
        hitTargetSelection
            .attr("x", node => node.x - node.hitRadius)
            .attr("y", node => node.y - node.hitRadius)
            .attr("rx", node => node.hitRadius)
            .attr("width", node => 2 * node.hitRadius)
            .attr("height", node => 2 * node.hitRadius);
        nodeSelection
            .attr("x", node => node.x - node.r)
            .attr("y", node => node.y - node.r)
            .attr("rx", node => node.r)
            .attr("width", node => 2 * node.r)
            .attr("height", node => 2 * node.r);
        mergedLabels.classed("search-hidden", hideClusterLabel);
    }

    function setHiddenLabels(predicate) {
        hideClusterLabel = predicate;
        clusterLabelLayer.selectAll("g.cluster-label").classed("search-hidden", hideClusterLabel);
    }

    const mapObserver = new ResizeObserver(render);
    mapObserver.observe(mapViewport);
    d3.select(window).on("resize.graph", render);
    return { publicationControls, render, setHiddenLabels };
}
