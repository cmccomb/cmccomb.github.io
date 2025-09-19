const svg = d3.select("svg");
const tooltip = d3.select(".tooltip");
const legendContainer = d3.select("#pub-legend");

d3.json("assets/json/pubs.json").then(data => {
    // Extents
    const xExtent = d3.extent(data, d => d.x);
    const yExtent = d3.extent(data, d => d.y);
    const yearExtent = d3.extent(data, d => d.pub_year);
    const rExtent = d3.extent(data, d => d.num_citations);

    const xScale = d3.scaleLinear().domain(xExtent);
    const yScale = d3.scaleLinear().domain(yExtent);
    const rScale = d3.scaleSqrt().domain(rExtent);
    const colorScale = d3.scaleSequential(d3.interpolateMagma).domain(yearExtent);

    const citationMedian = d3.median(data, d => d.num_citations);
    const citationEntries = [
        {label: "Min", value: rExtent[0]},
        {label: "Median", value: citationMedian},
        {label: "Max", value: rExtent[1]},
    ].filter(entry => Number.isFinite(entry.value));

    const citationLegendEntries = [];
    const seenCitationValues = new Set();
    citationEntries.forEach(entry => {
        const key = entry.value.toFixed(6);
        if (!seenCitationValues.has(key)) {
            seenCitationValues.add(key);
            citationLegendEntries.push(entry);
        }
    });
    citationLegendEntries.sort((a, b) => a.value - b.value);

    let colorLegendSvg = null;
    let colorLegendRect = null;
    let colorLegendAxis = null;
    let citationLegendSvg = null;
    const gradientId = "pub-legend-year-gradient";
    const citationLabelFormatter = d3.format(",");

    if (!legendContainer.empty()) {
        legendContainer.selectAll("*").remove();

        const colorLegend = legendContainer.append("div")
            .attr("class", "legend-section legend-section--color");
        colorLegend.append("span")
            .attr("class", "legend-title")
            .text("Publication year");

        colorLegendSvg = colorLegend.append("svg")
            .attr("class", "legend-colorbar");

        const colorDefs = colorLegendSvg.append("defs");
        const colorGradient = colorDefs.append("linearGradient")
            .attr("id", gradientId)
            .attr("x1", "0%")
            .attr("x2", "100%")
            .attr("y1", "0%")
            .attr("y2", "0%");

        const gradientStops = d3.range(0, 1.01, 0.1);
        colorGradient.selectAll("stop")
            .data(gradientStops)
            .enter()
            .append("stop")
            .attr("offset", d => `${(d * 100).toFixed(0)}%`)
            .attr("stop-color", d => colorScale(yearExtent[0] + d * (yearExtent[1] - yearExtent[0])));

        colorLegendRect = colorLegendSvg.append("rect")
            .attr("class", "legend-colorbar__swatch")
            .attr("fill", `url(#${gradientId})`);

        colorLegendAxis = colorLegendSvg.append("g")
            .attr("class", "legend-axis");

        if (citationLegendEntries.length > 0) {
            const citationLegend = legendContainer.append("div")
                .attr("class", "legend-section legend-section--citations");
            citationLegend.append("span")
                .attr("class", "legend-title")
                .text("Citations");

            citationLegendSvg = citationLegend.append("svg")
                .attr("class", "legend-citations");

            const citationGroups = citationLegendSvg.selectAll("g.legend-citation")
                .data(citationLegendEntries)
                .enter()
                .append("g")
                .attr("class", "legend-citation");

            citationGroups.append("rect")
                .attr("class", "legend-citation__swatch");

            citationGroups.append("text")
                .attr("class", "legend-citation__label");
        }
    }

    const nodes = data.map((d, i) => ({
        id: i,
        x_data: d.x,
        y_data: d.y,
        num_citations: d.num_citations,
        color: colorScale(d.pub_year),
        title: d.bib_dict.title,
        citation: d.bib_dict.citation,
        author: d.bib_dict.author,
        link: `https://scholar.google.com/citations?view_op=view_citation&citation_for_view=${d.author_pub_id}`,
        selected: false,
        x: 0,
        y: 0,
        x_orig: 0,
        y_orig: 0,
        r: 0,
    }));

    const nodeSel = svg.selectAll("rect")
        .data(nodes)
        .enter().append("rect")
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

        if (colorLegendSvg) {
            const legendWidth = Math.min(320, Math.max(200, width * 0.24));
            const colorBarHeight = width < 576 ? 12 : 14;

            colorLegendSvg
                .attr("width", legendWidth)
                .attr("height", colorBarHeight + 32);

            colorLegendRect
                .attr("width", legendWidth)
                .attr("height", colorBarHeight)
                .attr("rx", 6)
                .attr("ry", 6);

            const legendYearScale = d3.scaleLinear()
                .domain(yearExtent)
                .range([0, legendWidth]);

            const tickCount = width < 576 ? 2 : width < 992 ? 3 : 5;

            colorLegendAxis
                .attr("transform", `translate(0, ${colorBarHeight + 8})`)
                .call(
                    d3.axisBottom(legendYearScale)
                        .ticks(tickCount)
                        .tickSize(4)
                        .tickFormat(d3.format("d"))
                );
        }

        if (citationLegendSvg) {
            const legendWidth = Math.min(320, Math.max(200, width * 0.24));
            const citationGroups = citationLegendSvg.selectAll("g.legend-citation");
            const largestRadius = d3.max(citationLegendEntries, d => rScale(d.value)) || 0;
            const rowHeight = Math.max((largestRadius * 2) + 18, 36);

            citationLegendSvg
                .attr("width", legendWidth)
                .attr("height", rowHeight * citationLegendEntries.length);

            citationGroups
                .attr("transform", (d, i) => `translate(0, ${i * rowHeight})`);

            citationGroups.select("rect")
                .attr("width", d => 2 * rScale(d.value))
                .attr("height", d => 2 * rScale(d.value))
                .attr("rx", d => Math.min(12, rScale(d.value)))
                .attr("ry", d => Math.min(12, rScale(d.value)));

            citationGroups.select("text")
                .attr("x", d => (2 * rScale(d.value)) + 12)
                .attr("y", d => rScale(d.value))
                .attr("dominant-baseline", "middle")
                .text(d => {
                    const count = Math.max(0, Math.round(d.value));
                    const label = count === 1 ? "citation" : "citations";
                    return `${d.label}: ${citationLabelFormatter(count)} ${label}`;
                });
        }
    }

    render();

    d3.select(window).on("resize.graph", render);
    window.addEventListener("unload", () => {
        d3.select(window).on("resize.graph", null);
    });
});

