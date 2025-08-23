const svg = d3.select("svg");
const tooltip = d3.select(".tooltip");
const width = window.innerWidth;
const height = window.innerHeight;

svg
  .attr("width", width)
  .attr("height", height)
  .attr("viewBox", `0 0 ${width} ${height}`);
// .attr('preserveAspectRatio', 'none');

d3.json("assets/json/pubs.json").then((data) => {
  // 1) Scales
  const xExtent = d3.extent(data, (d) => d.x);
  const yExtent = d3.extent(data, (d) => d.y);
  const yearExtent = d3.extent(data, (d) => d.pub_year);
  const rExtent = d3.extent(data, (d) => d.num_citations);

  const xScale = d3
    .scaleLinear()
    .domain(xExtent)
    .range([40, width - 40]);
  const yScale = d3
    .scaleLinear()
    .domain(yExtent)
    .range([height - 40, 40]);
  const rScale = d3.scaleSqrt().domain(rExtent).range([10, 30]);
  const colorScale = d3.scaleSequential(d3.interpolateMagma).domain(yearExtent);

  // 2) Circle nodes
  const nodes = data.map((d, i) => ({
    id: i,
    x: xScale(d.x),
    y: yScale(d.y),
    x_orig: xScale(d.x),
    y_orig: yScale(d.y),
    r: rScale(d.num_citations),
    num_citations: d.num_citations,
    color: colorScale(d.pub_year),
    title: d.bib_dict.title,
    citation: d.bib_dict.citation,
    author: d.bib_dict.author,
    link: `https://scholar.google.com/citations?view_op=view_citation&citation_for_view=${d.author_pub_id}`,
    selected: false,
  }));

  // 3) Spread circles with a collide‐only force
  const circleSim = d3
    .forceSimulation(nodes)
    .force("x", d3.forceX((d) => d.x_orig).strength(1.0))
    .force("y", d3.forceY((d) => d.y_orig).strength(1.0))
    .force("collide", d3.forceCollide((d) => d.r + 1).strength(0.8))
    .stop();
  for (let i = 0; i < 200; ++i) circleSim.tick();

  // 4) Draw circles
  svg
    .selectAll("circle")
    .data(nodes)
    .enter()
    .append("circle")
    .attr("cx", (d) => d.x)
    .attr("cy", (d) => d.y)
    .attr("r", (d) => d.r)
    .attr("fill", (d) => d.color)
    .attr("opacity", 1.0)

    // Add a label box for each circle
    .on("mouseover", (e, d) => {
      // Make the mouse cursor a pointer
      e.currentTarget.style.cursor = "pointer";

      // Format the authors list
      let authors = d.author.split(" and ");
      if (authors.length > 2) {
        authors = authors.join(", ");
        // Do a final and for the last two authors
        authors = authors.replace(/, ([^,]*)$/, ", and $1");
      } else {
        authors = authors.join(" and ");
      }

      // Show tooltip
      tooltip
        .html(authors + '. "' + d.title + '." ' + d.citation + ".")
        .style("opacity", 1)
        .style("left", e.pageX + 10 + "px")
        .style("top", e.pageY + 10 + "px");
    })

    // Hide tooltip on mouse out
    .on("mouseout", () => {
      tooltip.style("opacity", 0);
    })

    // Open link to doubleclick
    .on("click", (e, d) => {
      window.open(d.link, "_blank");
    });

  // On window resize, update the position of the circles
  window.addEventListener("resize", () => {
    const newWidth = window.innerWidth;
    const newHeight = window.innerHeight;

    svg
      .attr("width", newWidth)
      .attr("height", newHeight)
      .attr("viewBox", `0 0 ${newWidth} ${newHeight}`);

    // Update positions of circles, as well as sizes and radii
    svg
      .selectAll("circle")
      .attr("cx", (d) => xScale(d.x_orig))
      .attr("cy", (d) => yScale(d.y_orig))
      .attr("r", (d) => d.r);
  });
});
