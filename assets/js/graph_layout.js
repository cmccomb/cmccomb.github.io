(() => {
    "use strict";

    const svg = d3.select("#publication-graph");
    const tooltip = d3.select(".tooltip");
    const legendContainer = d3.select(".colorbar-legend");
    const statusElement = document.getElementById("graph-status");
    const legacyLabelCorrections = new Map([
        ["face to face", "design teams"],
    ]);

    function showGraphError(error) {
        console.error("Unable to render publication graph", error);
        legendContainer.attr("hidden", true);
        if (statusElement) {
            statusElement.textContent = "The publication map is temporarily unavailable. Google Scholar is still available from the profile card.";
            statusElement.hidden = false;
        }
    }

    function asFiniteNumber(value) {
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    }

    function formatAuthors(authorValue) {
        const authors = String(authorValue || "").split(" and ").filter(Boolean);
        if (authors.length > 2) {
            const lastAuthor = authors.pop();
            return `${authors.join(", ")}, and ${lastAuthor}`;
        }
        return authors.join(" and ");
    }

    function displayClusterLabel(label) {
        const value = String(label || "").trim();
        return legacyLabelCorrections.get(value.toLowerCase()) || value;
    }

    d3.json("assets/json/pubs.json").then(rawPayload => {
        const rawRecords = Array.isArray(rawPayload?.records)
            ? rawPayload.records
            : Array.isArray(rawPayload)
                ? rawPayload
                : [];
        const clusters = Array.isArray(rawPayload?.clusters) ? rawPayload.clusters : [];
        const clusterSummaryById = new Map(
            clusters.map(cluster => [String(cluster.id), cluster])
        );

        const records = rawRecords.map(record => {
            const bibliography = record?.bib_dict;
            const x = asFiniteNumber(record?.x);
            const y = asFiniteNumber(record?.y);
            const publicationYear = asFiniteNumber(record?.pub_year);
            const citationCount = asFiniteNumber(record?.num_citations);
            const publicationId = typeof record?.author_pub_id === "string"
                ? record.author_pub_id
                : "";

            if (
                !bibliography
                || typeof bibliography.title !== "string"
                || x === null
                || y === null
                || publicationYear === null
                || citationCount === null
                || !publicationId
            ) {
                return null;
            }

            return {
                ...record,
                bib_dict: bibliography,
                x,
                y,
                pub_year: publicationYear,
                num_citations: Math.max(0, citationCount),
                author_pub_id: publicationId,
            };
        }).filter(Boolean);

        if (records.length === 0) {
            throw new Error("Publication snapshot contains no valid records");
        }

        const xExtent = d3.extent(records, record => record.x);
        const yExtent = d3.extent(records, record => record.y);
        const yearExtent = d3.extent(records, record => record.pub_year);
        const radiusExtent = d3.extent(records, record => record.num_citations);
        const xScale = d3.scaleLinear().domain(xExtent);
        const yScale = d3.scaleLinear().domain(yExtent);
        const radiusScale = d3.scaleSqrt().domain(radiusExtent);
        const colorScale = d3.scaleSequential(d3.interpolateMagma).domain(yearExtent);

        const legendWidth = 180;
        const legendHeight = 14;
        const legendMargins = { top: 20, right: 16, bottom: 30, left: 16 };
        const gradientId = "publication-year-gradient";
        const legendSvg = legendContainer.append("svg")
            .attr("class", "colorbar-svg")
            .attr("width", legendWidth + legendMargins.left + legendMargins.right)
            .attr("height", legendHeight + legendMargins.top + legendMargins.bottom);
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
        const tickCount = yearExtent[0] === yearExtent[1]
            ? 1
            : Math.min(6, Math.max(2, yearExtent[1] - yearExtent[0]));
        const legendAxis = d3.axisBottom(legendScale)
            .ticks(tickCount)
            .tickFormat(d3.format("d"));
        const axisGroup = legendSvg.append("g")
            .attr("transform", `translate(${legendMargins.left}, ${legendMargins.top + legendHeight})`)
            .call(legendAxis);

        axisGroup.selectAll("text")
            .attr("fill", "#f8f9fa")
            .attr("font-size", 10);
        axisGroup.selectAll("line, path")
            .attr("stroke", "rgba(248, 249, 250, 0.4)");
        legendSvg.append("text")
            .attr("x", legendMargins.left)
            .attr("y", legendMargins.top - 6)
            .attr("fill", "#f8f9fa")
            .attr("font-size", 12)
            .attr("font-weight", 600)
            .text("Publication Year");

        const nodes = records.map((record, index) => ({
            id: index,
            x_data: record.x,
            y_data: record.y,
            num_citations: record.num_citations,
            color: colorScale(record.pub_year),
            title: record.bib_dict.title,
            citation: String(record.bib_dict.citation || ""),
            author: String(record.bib_dict.author || ""),
            link: `https://scholar.google.com/citations?view_op=view_citation&citation_for_view=${encodeURIComponent(record.author_pub_id)}`,
            cluster_id: record.cluster_id,
            x: 0,
            y: 0,
            x_orig: 0,
            y_orig: 0,
            r: 0,
        }));

        const nodeSelection = svg.selectAll("rect.publication-node")
            .data(nodes)
            .enter()
            .append("rect")
            .attr("class", "publication-node")
            .attr("fill", node => node.color)
            .attr("opacity", 1)
            .attr("role", "link")
            .attr("aria-label", node => `${node.title}; ${node.num_citations} citations`)
            .on("mouseover", (event, node) => {
                event.currentTarget.style.cursor = "pointer";
                const tooltipText = [
                    formatAuthors(node.author),
                    `“${node.title}.”`,
                    node.citation,
                ].filter(Boolean).join(" ");
                tooltip.text(tooltipText)
                    .style("opacity", 1)
                    .style("left", `${event.pageX + 10}px`)
                    .style("top", `${event.pageY + 10}px`);
            })
            .on("mouseout", () => tooltip.style("opacity", 0))
            .on("click", (_event, node) => {
                window.open(node.link, "_blank", "noopener,noreferrer");
            });

        const clusterLabelLayer = svg.append("g")
            .attr("class", "cluster-label-layer")
            .attr("aria-hidden", "true");
        let simulation;

        function render() {
            const width = window.innerWidth;
            const height = window.innerHeight;
            svg.attr("width", width)
                .attr("height", height)
                .attr("viewBox", `0 0 ${width} ${height}`);

            xScale.range([40, width - 40]);
            yScale.range([height - 40, 40]);
            const radiusBase = Math.sqrt(width * height);
            radiusScale.range([radiusBase / 100, radiusBase / 50]);
            nodes.forEach(node => {
                node.x_orig = xScale(node.x_data);
                node.y_orig = yScale(node.y_data);
                node.r = radiusScale(node.num_citations);
            });

            if (!simulation) {
                simulation = d3.forceSimulation(nodes)
                    .force("x", d3.forceX(node => node.x_orig).strength(1))
                    .force("y", d3.forceY(node => node.y_orig).strength(1))
                    .force("collide", d3.forceCollide(node => node.r + 1).strength(0.8))
                    .stop();
            } else {
                simulation
                    .force("x", d3.forceX(node => node.x_orig).strength(1))
                    .force("y", d3.forceY(node => node.y_orig).strength(1))
                    .force("collide", d3.forceCollide(node => node.r + 1).strength(0.8));
            }

            simulation.alpha(1).restart();
            for (let index = 0; index < 200; index += 1) {
                simulation.tick();
            }
            nodeSelection
                .attr("x", node => node.x - node.r)
                .attr("y", node => node.y - node.r)
                .attr("rx", node => node.r)
                .attr("width", node => 2 * node.r)
                .attr("height", node => 2 * node.r);

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
                    const summary = clusterSummaryById.get(String(clusterId));
                    return {
                        clusterId,
                        label: typeof summary?.label === "string" && summary.label
                            ? displayClusterLabel(summary.label)
                            : `Cluster ${clusterId}`,
                        x: d3.mean(members, member => member.x) ?? 0,
                        y: d3.mean(members, member => member.y) ?? 0,
                    };
                }
            );
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
            mergedLabels.attr("transform", label => `translate(${label.x}, ${label.y})`);
            mergedLabels.select("text").text(label => label.label);
            mergedLabels.select("rect").each(function sizeLabelBackground() {
                const group = d3.select(this.parentNode);
                const textNode = group.select("text").node();
                if (!textNode) {
                    return;
                }
                const boundingBox = textNode.getBBox();
                const horizontalPadding = 12;
                const verticalPadding = 8;
                d3.select(this)
                    .attr("x", boundingBox.x - horizontalPadding / 2)
                    .attr("y", boundingBox.y - verticalPadding / 2)
                    .attr("width", boundingBox.width + horizontalPadding)
                    .attr("height", boundingBox.height + verticalPadding)
                    .attr("rx", 8)
                    .attr("ry", 8);
            });
        }

        render();
        d3.select(window).on("resize.graph", render);
        window.addEventListener("unload", () => {
            d3.select(window).on("resize.graph", null);
            simulation?.stop();
        });
    }).catch(showGraphError);
})();
