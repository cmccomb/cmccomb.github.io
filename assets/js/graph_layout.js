(() => {
    "use strict";

    const svg = d3.select("#publication-graph");
    const tooltip = d3.select(".tooltip");
    const legendContainer = d3.select(".colorbar-legend");
    const sizeLegendContainer = d3.select(".size-legend");
    const statusElement = document.getElementById("graph-status");
    const graphContainer = document.getElementById("graph-container");
    const graphCommandBar = document.getElementById("graph-command-bar");
    const searchInput = document.getElementById("publication-search");
    const searchClearButton = document.getElementById("publication-search-clear");
    const searchStatus = document.getElementById("publication-search-status");
    const detailPanel = document.getElementById("publication-detail");
    const detailCloseButton = document.getElementById("publication-detail-close");
    const detailTitle = document.getElementById("publication-detail-title");
    const detailMeta = document.getElementById("publication-detail-meta");
    const detailCitation = document.getElementById("publication-detail-citation");
    const detailLink = document.getElementById("publication-detail-link");
    const legacyLabelCorrections = new Map([
        ["face to face", "design teams"],
    ]);

    function showGraphError(error) {
        console.error("Unable to render publication graph", error);
        legendContainer.attr("hidden", true);
        sizeLegendContainer.attr("hidden", true);
        if (searchInput) {
            searchInput.disabled = true;
        }
        if (searchClearButton) {
            searchClearButton.disabled = true;
        }
        if (searchStatus) {
            searchStatus.textContent = "Publication search is unavailable.";
        }
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
        const sizeLegendHeight = 72;
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
            const positions = values.length === 1 ? [70] : [20, 70, 118];
            const legendEntries = values.map((value, index) => ({
                value,
                x: positions[index],
                radius: radiusScale(value),
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
            sizeKeys.attr("transform", entry => `translate(${entry.x}, 32)`);
            sizeKeys.select("circle").attr("r", entry => entry.radius);
            sizeKeys.select("text")
                .attr("y", 34)
                .text(entry => d3.format(",d")(Math.round(entry.value)));
            sizeLegendContainer.attr(
                "aria-label",
                `Circle sizes represent citation counts from ${Math.round(radiusExtent[0])} to ${Math.round(radiusExtent[1])}.`
            );
        }

        const nodes = records.map((record, index) => {
            const clusterSummary = clusterSummaryById.get(String(record.cluster_id));
            const clusterLabel = typeof clusterSummary?.label === "string" && clusterSummary.label
                ? displayClusterLabel(clusterSummary.label)
                : `Cluster ${record.cluster_id}`;
            const title = record.bib_dict.title;
            const citation = String(record.bib_dict.citation || "");
            const author = String(record.bib_dict.author || "");
            const abstract = String(record.bib_dict.abstract || "");

            return {
                id: index,
                x_data: record.x,
                y_data: record.y,
                pub_year: record.pub_year,
                num_citations: record.num_citations,
                color: colorScale(record.pub_year),
                title,
                citation,
                author,
                abstract,
                link: `https://scholar.google.com/citations?view_op=view_citation&citation_for_view=${encodeURIComponent(record.author_pub_id)}`,
                cluster_id: record.cluster_id,
                cluster_label: clusterLabel,
                search_text: [
                    title,
                    author,
                    citation,
                    abstract,
                    record.pub_year,
                    clusterLabel,
                ].join(" ").toLowerCase(),
                x: 0,
                y: 0,
                x_orig: 0,
                y_orig: 0,
                r: 0,
            };
        });

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
                `${node.title}; ${node.pub_year}; ${node.num_citations} citation${node.num_citations === 1 ? "" : "s"}; topic ${node.cluster_label}; show publication details`
            ));
        const nodeSelection = publicationControls.append("rect")
            .attr("class", "publication-node")
            .attr("fill", node => node.color)
            .attr("opacity", 1);
        const publicationControlNodes = publicationControls.nodes();
        let matchingPublicationIndices = nodes.map((_node, index) => index);
        let activePublicationIndex = 0;
        let selectedPublicationIndex = null;
        let clusterLabelLayer;

        function graphIsActive() {
            return graphContainer?.classList.contains("graph-active") ?? false;
        }

        function setRovingIndex(index, shouldFocus = false) {
            if (matchingPublicationIndices.length === 0) {
                activePublicationIndex = -1;
                publicationControls.attr("tabindex", -1);
                return;
            }

            activePublicationIndex = matchingPublicationIndices.includes(index)
                ? index
                : matchingPublicationIndices[0];
            publicationControls.attr("tabindex", (_node, controlIndex) => (
                graphIsActive() && controlIndex === activePublicationIndex ? 0 : -1
            ));

            if (shouldFocus) {
                publicationControlNodes[activePublicationIndex]?.focus({ preventScroll: true });
            }
        }

        function setDetailVisibility(isVisible) {
            if (!detailPanel) {
                return;
            }
            detailPanel.hidden = !isVisible;
            detailPanel.setAttribute("aria-hidden", String(!isVisible));
            graphContainer?.classList.toggle("detail-active", isVisible);
        }

        function clearPublicationDetail({ restoreFocus = false } = {}) {
            const previouslySelectedIndex = selectedPublicationIndex;
            selectedPublicationIndex = null;
            publicationControls
                .classed("selected", false)
                .attr("aria-expanded", "false");
            setDetailVisibility(false);

            if (detailTitle) {
                detailTitle.textContent = "";
            }
            if (detailMeta) {
                detailMeta.textContent = "";
            }
            if (detailCitation) {
                detailCitation.textContent = "";
            }
            if (detailLink) {
                detailLink.setAttribute("href", "#");
            }

            if (
                restoreFocus
                && graphIsActive()
                && previouslySelectedIndex !== null
                && matchingPublicationIndices.includes(previouslySelectedIndex)
            ) {
                setRovingIndex(previouslySelectedIndex, true);
            }
        }

        function showPublicationDetail(index, node) {
            selectedPublicationIndex = index;
            publicationControls
                .classed("selected", (_publication, controlIndex) => controlIndex === index)
                .attr("aria-expanded", (_publication, controlIndex) => (
                    controlIndex === index ? "true" : "false"
                ));

            if (detailTitle) {
                detailTitle.textContent = node.title;
            }
            if (detailMeta) {
                detailMeta.textContent = `${node.pub_year} · ${node.num_citations} citation${node.num_citations === 1 ? "" : "s"} · Topic: ${node.cluster_label}`;
            }
            if (detailCitation) {
                detailCitation.textContent = [
                    formatAuthors(node.author),
                    node.citation,
                ].filter(Boolean).join(". ");
            }
            if (detailLink) {
                detailLink.setAttribute("href", node.link);
            }

            hideTooltip();
            setDetailVisibility(true);
        }

        function updateClusterLabelVisibility() {
            if (!clusterLabelLayer) {
                return;
            }
            const visibleClusterIds = new Set(
                matchingPublicationIndices.map(index => String(nodes[index].cluster_id))
            );
            const hasQuery = Boolean(searchInput?.value.trim());
            clusterLabelLayer.selectAll("g.cluster-label")
                .classed("search-hidden", label => (
                    hasQuery && !visibleClusterIds.has(String(label.clusterId))
                ));
        }

        function applySearch(query) {
            const normalizedQuery = String(query || "").trim().toLowerCase();
            const searchTerms = normalizedQuery.split(/\s+/).filter(Boolean);
            matchingPublicationIndices = nodes
                .map((node, index) => ({ node, index }))
                .filter(({ node }) => searchTerms.every(term => node.search_text.includes(term)))
                .map(({ index }) => index);
            const matchingSet = new Set(matchingPublicationIndices);
            const hasQuery = searchTerms.length > 0;

            publicationControls
                .classed("search-hidden", (_node, index) => !matchingSet.has(index))
                .classed("search-match", (_node, index) => hasQuery && matchingSet.has(index))
                .attr("aria-hidden", (_node, index) => (
                    matchingSet.has(index) ? null : "true"
                ));

            if (
                selectedPublicationIndex !== null
                && !matchingSet.has(selectedPublicationIndex)
            ) {
                clearPublicationDetail();
            }

            setRovingIndex(activePublicationIndex);
            updateClusterLabelVisibility();

            if (searchClearButton) {
                searchClearButton.disabled = String(query || "").length === 0;
            }
            if (searchStatus) {
                if (!hasQuery) {
                    searchStatus.textContent = `${nodes.length} publications`;
                } else {
                    const count = matchingPublicationIndices.length;
                    searchStatus.textContent = `${count} of ${nodes.length} publications`;
                }
            }
        }

        function getTooltipText(node) {
            return [
                formatAuthors(node.author),
                `“${node.title}.”`,
                node.citation,
            ].filter(Boolean).join(" ");
        }

        function positionTooltip(clientX, clientY) {
            const tooltipNode = tooltip.node();
            if (!tooltipNode) {
                return;
            }

            const margin = 12;
            const offset = 12;
            const maxLeft = Math.max(margin, window.innerWidth - tooltipNode.offsetWidth - margin);
            const maxTop = Math.max(margin, window.innerHeight - tooltipNode.offsetHeight - margin);
            const preferredTop = clientY + offset + tooltipNode.offsetHeight <= window.innerHeight - margin
                ? clientY + offset
                : clientY - tooltipNode.offsetHeight - offset;

            tooltip
                .style("left", `${Math.max(margin, Math.min(clientX + offset, maxLeft))}px`)
                .style("top", `${Math.max(margin, Math.min(preferredTop, maxTop))}px`);
        }

        function showPointerTooltip(event, node) {
            if (detailPanel && !detailPanel.hidden) {
                return;
            }
            tooltip
                .text(getTooltipText(node))
                .attr("aria-hidden", "false")
                .style("opacity", 1);
            positionTooltip(event.clientX, event.clientY);
        }

        function showFocusTooltip(event, node) {
            if (detailPanel && !detailPanel.hidden) {
                return;
            }
            const target = event.currentTarget.querySelector(".publication-node")
                || event.currentTarget;
            const bounds = target.getBoundingClientRect();
            tooltip
                .text(getTooltipText(node))
                .attr("aria-hidden", "false")
                .style("opacity", 1);
            positionTooltip(
                bounds.left + bounds.width / 2,
                bounds.top + bounds.height / 2
            );
        }

        function hideTooltip() {
            tooltip
                .attr("aria-hidden", "true")
                .style("opacity", 0);
        }

        publicationControls
            .on("mousemove", showPointerTooltip)
            .on("mouseleave", event => {
                if (document.activeElement !== event.currentTarget) {
                    hideTooltip();
                }
            })
            .on("focus", function handlePublicationFocus(event, node) {
                activePublicationIndex = publicationControlNodes.indexOf(this);
                setRovingIndex(activePublicationIndex);
                showFocusTooltip(event, node);
            })
            .on("blur", hideTooltip)
            .on("click", function handlePublicationClick(event, node) {
                const isPlainPrimaryClick = (
                    event.button === 0
                    && !event.altKey
                    && !event.ctrlKey
                    && !event.metaKey
                    && !event.shiftKey
                );
                if (!isPlainPrimaryClick) {
                    return;
                }

                event.preventDefault();
                const index = publicationControlNodes.indexOf(this);
                setRovingIndex(index);
                showPublicationDetail(index, node);
            })
            .on("keydown", function handlePublicationKeydown(event, node) {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    const index = publicationControlNodes.indexOf(this);
                    setRovingIndex(index);
                    showPublicationDetail(index, node);
                    return;
                }

                if (matchingPublicationIndices.length === 0) {
                    return;
                }
                const currentPosition = matchingPublicationIndices.indexOf(
                    activePublicationIndex
                );
                let nextIndex;
                switch (event.key) {
                    case "ArrowRight":
                    case "ArrowDown":
                        nextIndex = matchingPublicationIndices[
                            (currentPosition + 1) % matchingPublicationIndices.length
                        ];
                        break;
                    case "ArrowLeft":
                    case "ArrowUp":
                        nextIndex = matchingPublicationIndices[
                            (currentPosition - 1 + matchingPublicationIndices.length)
                            % matchingPublicationIndices.length
                        ];
                        break;
                    case "Home":
                        nextIndex = matchingPublicationIndices[0];
                        break;
                    case "End":
                        nextIndex = matchingPublicationIndices.at(-1);
                        break;
                    default:
                        return;
                }

                event.preventDefault();
                setRovingIndex(nextIndex, true);
            });

        searchInput?.addEventListener("input", () => {
            applySearch(searchInput.value);
        });
        searchInput?.addEventListener("keydown", event => {
            if (event.key !== "Enter") {
                return;
            }

            event.preventDefault();
            if (matchingPublicationIndices.length > 0) {
                setRovingIndex(matchingPublicationIndices[0], true);
            }
        });
        searchClearButton?.addEventListener("click", () => {
            if (!searchInput) {
                return;
            }
            searchInput.value = "";
            applySearch("");
            searchInput.focus({ preventScroll: true });
        });
        detailCloseButton?.addEventListener("click", () => {
            clearPublicationDetail({ restoreFocus: true });
        });

        graphContainer?.addEventListener("publicationgraph:visibilitychange", event => {
            if (event.detail?.isVisible) {
                setRovingIndex(activePublicationIndex);
                render();
                return;
            }

            if (searchInput) {
                searchInput.value = "";
            }
            activePublicationIndex = 0;
            applySearch("");
            clearPublicationDetail();
            publicationControls.attr("tabindex", -1);
            hideTooltip();
        });

        clusterLabelLayer = svg.append("g")
            .attr("class", "cluster-label-layer")
            .attr("aria-hidden", "true");
        let simulation;

        function render() {
            const containerBounds = graphContainer?.getBoundingClientRect();
            const width = Math.max(1, Math.round(containerBounds?.width || window.innerWidth));
            const height = Math.max(1, Math.round(containerBounds?.height || window.innerHeight));
            svg.attr("width", width)
                .attr("height", height)
                .attr("viewBox", `0 0 ${width} ${height}`);

            const mapPadding = width <= 768 ? 24 : 40;
            const commandBarBounds = graphCommandBar && !graphCommandBar.hidden
                ? graphCommandBar.getBoundingClientRect()
                : null;
            const topPadding = commandBarBounds
                ? Math.min(
                    height - mapPadding,
                    Math.max(mapPadding, commandBarBounds.bottom + 16)
                )
                : mapPadding;
            const bottomCoordinate = Math.max(topPadding + 1, height - mapPadding);
            xScale.range([mapPadding, width - mapPadding]);
            yScale.range([bottomCoordinate, topPadding]);
            const radiusBase = Math.sqrt(width * height);
            radiusScale.range([12.1, Math.min(20, Math.max(16, radiusBase / 60))]);
            renderSizeLegend();
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

            simulation.alpha(1).stop();
            const constrainNodeToMap = node => {
                const minimumX = node.r;
                const maximumX = width - node.r;
                const minimumY = topPadding + node.r;
                const maximumY = bottomCoordinate - node.r;
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
            mergedLabels.select("text").text(label => label.label);
            mergedLabels.each(function sizeAndPositionLabel(label) {
                const group = d3.select(this);
                const textNode = group.select("text").node();
                if (!textNode) {
                    return;
                }
                const boundingBox = textNode.getBBox();
                const horizontalPadding = 12;
                const verticalPadding = 8;
                const labelWidth = boundingBox.width + horizontalPadding;
                const labelHeight = boundingBox.height + verticalPadding;
                const labelMargin = 8;
                const clampedX = Math.max(
                    labelMargin + labelWidth / 2,
                    Math.min(width - labelMargin - labelWidth / 2, label.x)
                );
                const clampedY = Math.max(
                    labelMargin + labelHeight / 2,
                    Math.min(height - labelMargin - labelHeight / 2, label.y)
                );

                group
                    .attr("transform", `translate(${clampedX}, ${clampedY})`)
                    .select("rect")
                    .attr("x", boundingBox.x - horizontalPadding / 2)
                    .attr("y", boundingBox.y - verticalPadding / 2)
                    .attr("width", labelWidth)
                    .attr("height", labelHeight)
                    .attr("rx", 8)
                    .attr("ry", 8);
            });
            updateClusterLabelVisibility();
        }

        render();
        if (searchInput) {
            searchInput.disabled = false;
        }
        applySearch("");
        d3.select(window).on("resize.graph", render);
        window.addEventListener("unload", () => {
            d3.select(window).on("resize.graph", null);
            simulation?.stop();
        });
    }).catch(showGraphError);
})();
