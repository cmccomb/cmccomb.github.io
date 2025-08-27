const svg = d3.select("svg");
const tooltip = d3.select(".tooltip");

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
    }

    render();

    d3.select(window).on("resize.graph", render);
    window.addEventListener("unload", () => {
        d3.select(window).on("resize.graph", null);
    });
});

