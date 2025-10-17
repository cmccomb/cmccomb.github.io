const svg = d3.select("svg");
const tooltip = d3.select(".tooltip");
const legendContainer = d3.select(".colorbar-legend");

d3.json("assets/json/pubs.json").then(rawPayload => {
    const records = Array.isArray(rawPayload?.records)
        ? rawPayload.records
        : Array.isArray(rawPayload)
            ? rawPayload
            : [];
    const clusters = Array.isArray(rawPayload?.clusters) ? rawPayload.clusters : [];
    const clusterSummaryById = new Map(
        clusters.map(cluster => [String(cluster.id), cluster])
    );

    // Extents
    const xExtent = d3.extent(records, d => d.x);
    const yExtent = d3.extent(records, d => d.y);
    const yearExtent = d3.extent(records, d => d.pub_year);
    const rExtent = d3.extent(records, d => d.num_citations);

    const xScale = d3.scaleLinear().domain(xExtent);
    const yScale = d3.scaleLinear().domain(yExtent);
    const rScale = d3.scaleSqrt().domain(rExtent);
    const colorScale = d3.scaleSequential(d3.interpolateMagma).domain(yearExtent);

    const legendWidth = 180;
    const legendHeight = 14;
    const legendMargins = { top: 20, right: 16, bottom: 30, left: 16 };
    const gradientId = "publication-year-gradient";

    if (!legendContainer.select("svg").node()) {
        const legendSvg = legendContainer
            .append("svg")
            .attr("class", "colorbar-svg")
            .attr("width", legendWidth + legendMargins.left + legendMargins.right)
            .attr("height", legendHeight + legendMargins.top + legendMargins.bottom);

        const defs = legendSvg.append("defs");
        const gradient = defs
            .append("linearGradient")
            .attr("id", gradientId)
            .attr("x1", "0%")
            .attr("x2", "100%")
            .attr("y1", "0%")
            .attr("y2", "0%");

        const gradientStops = d3.range(0, 1.0001, 0.05);
        gradient
            .selectAll("stop")
            .data(gradientStops)
            .enter()
            .append("stop")
            .attr("offset", d => `${d * 100}%`)
            .attr("stop-color", d => colorScale(yearExtent[0] + d * (yearExtent[1] - yearExtent[0])));

        legendSvg
            .append("rect")
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

        const axisGroup = legendSvg
            .append("g")
            .attr("transform", `translate(${legendMargins.left}, ${legendMargins.top + legendHeight})`)
            .call(legendAxis);

        axisGroup.selectAll("text")
            .attr("fill", "#f8f9fa")
            .attr("font-size", 10);

        axisGroup.selectAll("line")
            .attr("stroke", "rgba(248, 249, 250, 0.4)");

        axisGroup.selectAll("path")
            .attr("stroke", "rgba(248, 249, 250, 0.4)");

        legendSvg
            .append("text")
            .attr("x", legendMargins.left)
            .attr("y", legendMargins.top - 6)
            .attr("fill", "#f8f9fa")
            .attr("font-size", 12)
            .attr("font-weight", 600)
            .text("Publication Year");
    }

    const nodes = records.map((d, i) => ({
        id: i,
        x_data: d.x,
        y_data: d.y,
        num_citations: d.num_citations,
        color: colorScale(d.pub_year),
        title: d.bib_dict.title,
        citation: d.bib_dict.citation,
        author: d.bib_dict.author,
        link: `https://scholar.google.com/citations?view_op=view_citation&citation_for_view=${d.author_pub_id}`,
        cluster_id: d.cluster_id,
        selected: false,
        x: 0,
        y: 0,
        x_orig: 0,
        y_orig: 0,
        r: 0,
    }));

    const nodeSel = svg.selectAll("rect.publication-node")
        .data(nodes)
        .enter().append("rect")
        .attr("class", "publication-node")
        .attr("fill", d => d.color)
        .attr("opacity", 1.0)
        .on("mouseover", (e, d) => {
            e.currentTarget.style.cursor = "pointer";

            let authors = d.author.split(" and ");
            if (authors.length > 2) {
                authors = authors.join(", ");
                authors = authors.replace(/, ([^,]*)$/, ", and $1");
            } else {
                authors = authors.join(" and ");
            }

            tooltip
                .html(authors + '. "' + d.title + '." ' + d.citation + '.')
                .style("opacity", 1)
                .style("left", (e.pageX + 10) + "px")
                .style("top", (e.pageY + 10) + "px");
        })
        .on("mouseout", () => {
            tooltip.style("opacity", 0);
        })
        .on("click", (e, d) => {
            window.open(d.link, "_blank");
        });

    const clusterLabelLayer = svg
        .append("g")
        .attr("class", "cluster-label-layer")
        .attr("aria-hidden", "true");

    let circleSim;

    function render() {
        const width = window.innerWidth;
        const height = window.innerHeight;

        svg
            .attr('width', width)
            .attr('height', height)
            .attr('viewBox', `0 0 ${width} ${height}`);

        xScale.range([40, width - 40]);
        yScale.range([height - 40, 40]);

        const radiusBase = Math.sqrt(width * height);
        rScale.range([radiusBase / 100, radiusBase / 50]);

        nodes.forEach(d => {
            d.x_orig = xScale(d.x_data);
            d.y_orig = yScale(d.y_data);
            d.r = rScale(d.num_citations);
        });

        if (!circleSim) {
            circleSim = d3.forceSimulation(nodes)
                .force("x", d3.forceX(d => d.x_orig).strength(1.0))
                .force("y", d3.forceY(d => d.y_orig).strength(1.0))
                .force("collide", d3.forceCollide(d => d.r + 1).strength(0.8))
                .stop();
        } else {
            circleSim
                .force("x", d3.forceX(d => d.x_orig).strength(1.0))
                .force("y", d3.forceY(d => d.y_orig).strength(1.0))
                .force("collide", d3.forceCollide(d => d.r + 1).strength(0.8));
        }

        circleSim.alpha(1).restart();
        for (let i = 0; i < 200; ++i) circleSim.tick();

        nodeSel
            .attr("x", d => d.x - d.r)
            .attr("y", d => d.y - d.r)
            .attr("rx", d => d.r)
            .attr("width", d => 2 * d.r)
            .attr("height", d => 2 * d.r);

        const clusteredNodes = nodes.filter(d => (
            d.cluster_id !== null && d.cluster_id !== undefined && Number.isFinite(d.x) && Number.isFinite(d.y)
        ));

        if (clusteredNodes.length === 0) {
            clusterLabelLayer.selectAll("g.cluster-label").remove();
            return;
        }

        const labelData = Array.from(
            d3.group(clusteredNodes, d => d.cluster_id),
            ([clusterId, members]) => {
                const summary = clusterSummaryById.get(String(clusterId));
                return {
                    clusterId,
                    label: summary?.label ?? `Cluster ${clusterId}`,
                    x: d3.mean(members, member => member.x) ?? 0,
                    y: d3.mean(members, member => member.y) ?? 0,
                };
            }
        );

        const mergedLabels = clusterLabelLayer
            .selectAll("g.cluster-label")
            .data(labelData, d => d.clusterId)
            .join(enter => {
                const group = enter.append("g").attr("class", "cluster-label");
                group.append("rect").attr("class", "cluster-label-background");
                group.append("text")
                    .attr("class", "cluster-label-text")
                    .attr("text-anchor", "middle")
                    .attr("dominant-baseline", "middle");
                return group;
            });

        mergedLabels
            .attr("transform", d => `translate(${d.x}, ${d.y})`);

        mergedLabels
            .select("text")
            .text(d => d.label);

        mergedLabels.select("rect")
            .each(function () {
                const group = d3.select(this.parentNode);
                const textNode = group.select("text").node();
                if (!textNode) {
                    return;
                }

                const bbox = textNode.getBBox();
                const horizontalPadding = 12;
                const verticalPadding = 8;

                d3.select(this)
                    .attr("x", bbox.x - horizontalPadding / 2)
                    .attr("y", bbox.y - verticalPadding / 2)
                    .attr("width", bbox.width + horizontalPadding)
                    .attr("height", bbox.height + verticalPadding)
                    .attr("rx", 8)
                    .attr("ry", 8);
            });
    }

    render();

    d3.select(window).on("resize.graph", render);
    window.addEventListener("unload", () => {
        d3.select(window).on("resize.graph", null);
    });
});

