(() => {
    "use strict";

    const svg = d3.select("#publication-graph");
    const tooltip = d3.select(".tooltip");
    const legendContainer = d3.select(".colorbar-legend");
    const sizeLegendContainer = d3.select(".size-legend");
    const statusElement = document.getElementById("graph-status");
    const statusMessage = document.getElementById("graph-status-message");
    const statusDismiss = document.getElementById("graph-status-dismiss");
    const statusScholar = document.getElementById("graph-status-scholar");
    const graphContainer = document.getElementById("graph-container");
    const graphCommandBar = document.getElementById("graph-command-bar");
    const searchInput = document.getElementById("publication-search");
    const searchClearButton = document.getElementById("publication-search-clear");
    const searchStatus = document.getElementById("publication-search-status");
    const detailPanel = document.getElementById("publication-detail");
    const detailBody = document.getElementById("publication-detail-body");
    const mapViewport = document.getElementById("publication-map-viewport");
    const detailCloseButton = document.getElementById("publication-detail-close");
    const detailTitle = document.getElementById("publication-detail-title");
    const detailMeta = document.getElementById("publication-detail-meta");
    const detailCitation = document.getElementById("publication-detail-citation");
    const detailLink = document.getElementById("publication-detail-link");
    const profileTitle = document.title;
    const helpers = window.PublicationHelpers;
    const resultsPanel = document.getElementById("publication-results");
    const resultsList = document.getElementById("publication-list");
    const emptyResults = document.getElementById("publication-empty");
    const viewButtons = {
        list: document.getElementById("publication-view-list"),
        map: document.getElementById("publication-view-map"),
    };
    const resourcesPanel = document.getElementById("publication-resources");
    const copyStatus = document.getElementById("publication-copy-status");
    const copyFallback = document.getElementById("publication-copy-fallback");
    const copyFallbackLabel = document.getElementById("publication-copy-fallback-label");
    const legacyLabelCorrections = new Map([
        ["face to face", "design teams"],
    ]);

    function updateToolbarGeometry() {
        const bottom = graphCommandBar.getBoundingClientRect().bottom;
        graphContainer.style.setProperty("--browser-content-top", `${bottom + 12}px`);
    }
    const toolbarObserver = new ResizeObserver(updateToolbarGeometry);
    toolbarObserver.observe(graphCommandBar);
    statusDismiss.addEventListener("click", () => {
        statusElement.hidden = true;
        searchInput.focus({ preventScroll: true });
    });

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
            graphContainer.classList.add("data-unavailable");
            resultsPanel.hidden = true;
            emptyResults.hidden = true;
            Object.values(viewButtons).forEach(button => {
                button.disabled = true;
                button.setAttribute("aria-pressed", "false");
            });
            statusMessage.textContent = "The publication browser is temporarily unavailable.";
            statusDismiss.hidden = true;
            statusScholar.hidden = false;
            statusElement.hidden = false;
            updateToolbarGeometry();
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
                record,
                publicationId: record.author_pub_id,
                x_data: record.x,
                y_data: record.y,
                pub_year: record.pub_year,
                num_citations: record.num_citations,
                color: colorScale(record.pub_year),
                title,
                citation: helpers.venue(record),
                author,
                abstract,
                link: `https://scholar.google.com/citations?view_op=view_citation&citation_for_view=${encodeURIComponent(record.author_pub_id)}`,
                cluster_id: record.cluster_id,
                cluster_label: clusterLabel,
                searchIndex: helpers.searchIndex([
                    title,
                    author,
                    citation,
                    abstract,
                    record.pub_year,
                    clusterLabel,
                ], title),
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
        let currentView = window.matchMedia("(max-width: 768px)").matches ? "list" : "map";
        let restoringLocation = false;
        const listItems = nodes.map((node, index) => {
            const item = document.createElement("li");
            const button = document.createElement("button");
            button.type = "button";
            button.className = "publication-result";
            button.dataset.publicationId = node.publicationId;
            button.style.setProperty("--publication-accent", node.color);
            button.setAttribute("aria-controls", "publication-detail");
            button.setAttribute("aria-expanded", "false");
            const title = document.createElement("span");
            title.className = "publication-result-title";
            title.textContent = node.title;
            const authors = document.createElement("span");
            authors.className = "publication-result-authors";
            authors.textContent = formatAuthors(node.author);
            const meta = document.createElement("span");
            meta.className = "publication-result-meta";
            meta.textContent = `${node.pub_year} · ${node.cluster_label} · ${node.num_citations} citation${node.num_citations === 1 ? "" : "s"}`;
            button.append(title, authors, meta);
            button.addEventListener("click", () => {
                setRovingIndex(index);
                showPublicationDetail(index, node);
                detailTitle.focus({ preventScroll: true });
            });
            item.append(button);
            return { item, button };
        });

        function writeLocation({ replace = false } = {}) {
            if (restoringLocation || !graphIsActive()) return;
            const url = new URL(window.location.href);
            url.searchParams.set("view", currentView);
            if (searchInput.value) url.searchParams.set("q", searchInput.value);
            else url.searchParams.delete("q");
            if (selectedPublicationIndex !== null) {
                url.searchParams.set("paper", nodes[selectedPublicationIndex].publicationId);
            } else url.searchParams.delete("paper");
            if (url.href !== window.location.href) {
                window.history[replace ? "replaceState" : "pushState"](null, "", url);
            }
        }

        function renderList() {
            const visible = new Set(matchingPublicationIndices);
            listItems.forEach(({ item, button }, index) => {
                item.hidden = !visible.has(index);
                button.setAttribute("aria-expanded", String(index === selectedPublicationIndex));
            });
            const sorted = [...matchingPublicationIndices].sort((a, b) => (
                helpers.searchScore(nodes[b].searchIndex, searchInput.value)
                - helpers.searchScore(nodes[a].searchIndex, searchInput.value)
                || nodes[b].pub_year - nodes[a].pub_year
                || nodes[b].num_citations - nodes[a].num_citations
                || nodes[a].title.localeCompare(nodes[b].title)
            ));
            sorted.forEach(index => resultsList.append(listItems[index].item));
            resultsPanel.querySelector(".publication-results-help").textContent = helpers.normalize(searchInput.value)
                ? "Best matches first. Select a publication for details and links."
                : "Newest first. Search to find a title, author, topic, or year.";
            resultsPanel.querySelector(".publication-results-help").hidden = sorted.length === 0;
        }

        function setView(view, { save = true } = {}) {
            currentView = view === "list" ? "list" : "map";
            const layoutChanged = graphContainer.classList.contains("list-view") !== (currentView === "list");
            graphContainer.classList.toggle("list-view", currentView === "list");
            resultsPanel.hidden = currentView !== "list" && (!graphIsActive() || matchingPublicationIndices.length > 0);
            svg.attr("aria-hidden", currentView === "list" ? "true" : null);
            Object.entries(viewButtons).forEach(([name, button]) => {
                button.setAttribute("aria-pressed", String(name === currentView));
            });
            setRovingIndex(activePublicationIndex);
            hideTooltip();
            if (layoutChanged) render();
            if (save) {
                graphContainer.dispatchEvent(new CustomEvent("publicationgraph:viewchange", {
                    detail: { view: currentView },
                }));
                writeLocation();
            }
        }

        function restorePublicationLocation() {
            restoringLocation = true;
            const params = new URLSearchParams(window.location.search);
            searchInput.value = graphIsActive() ? (params.get("q") || "") : "";
            clearPublicationDetail();
            applySearch(searchInput.value);
            setView(graphContainer.dataset.publicationView || (window.matchMedia("(max-width: 768px)").matches ? "list" : "map"), { save: false });
            const id = params.get("paper");
            const index = nodes.findIndex(node => node.publicationId === id);
            statusElement.hidden = true;
            if (graphIsActive() && id) {
                if (index >= 0) {
                    setRovingIndex(index);
                    showPublicationDetail(index, nodes[index]);
                    detailTitle.focus({ preventScroll: true });
                } else {
                    statusMessage.textContent = "This publication is not in the current collection. Search or browse the publications below.";
                    statusElement.hidden = false;
                }
            }
            restoringLocation = false;
        }

        function resetCopyFeedback() {
            copyStatus.textContent = "";
            copyFallback.hidden = true;
            copyFallbackLabel.hidden = true;
            copyFallback.value = "";
        }

        async function copyText(text, successMessage) {
            const copiedIndex = selectedPublicationIndex;
            resetCopyFeedback();
            try {
                await navigator.clipboard.writeText(text);
                if (selectedPublicationIndex !== copiedIndex) return;
                copyStatus.textContent = successMessage;
                copyStatus.scrollIntoView({ block: "nearest" });
            } catch {
                if (selectedPublicationIndex !== copiedIndex) return;
                copyStatus.textContent = "Automatic copy is unavailable. Select and copy the text below.";
                copyFallback.value = text;
                copyFallback.hidden = false;
                copyFallbackLabel.hidden = false;
                copyFallback.focus();
                copyFallback.select();
            }
        }
        document.getElementById("publication-copy-link").addEventListener("click", () => {
            if (selectedPublicationIndex === null) return;
            const url = new URL(window.location.pathname, window.location.origin);
            url.searchParams.set("view", currentView);
            if (searchInput.value) url.searchParams.set("q", searchInput.value);
            url.searchParams.set("paper", nodes[selectedPublicationIndex].publicationId);
            copyText(url.href, "Publication link copied.");
        });
        document.getElementById("publication-copy-citation").addEventListener("click", () => {
            if (selectedPublicationIndex !== null) {
                copyText(helpers.citation(nodes[selectedPublicationIndex].record), "Citation copied.");
            }
        });
        document.getElementById("publication-empty-clear").addEventListener("click", () => searchClearButton.click());
        Object.entries(viewButtons).forEach(([view, button]) => {
            button.addEventListener("click", () => setView(view));
        });

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
                graphIsActive() && currentView === "map" && controlIndex === activePublicationIndex ? 0 : -1
            ));

            if (shouldFocus) {
                const target = currentView === "list" ? listItems[activePublicationIndex]?.button
                    : publicationControlNodes[activePublicationIndex];
                target?.focus({ preventScroll: currentView !== "list" });
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
            document.title = profileTitle;
            const previouslySelectedIndex = selectedPublicationIndex;
            selectedPublicationIndex = null;
            publicationControls
                .classed("selected", false)
                .attr("aria-expanded", "false");
            setDetailVisibility(false);
            listItems.forEach(({ button }) => button.setAttribute("aria-expanded", "false"));
            resourcesPanel.replaceChildren();
            resetCopyFeedback();

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
            if (restoreFocus) writeLocation();
        }

        function showPublicationDetail(index, node) {
            statusElement.hidden = true;
            document.title = `${node.title} — ${profileTitle}`;
            selectedPublicationIndex = index;
            resetCopyFeedback();
            listItems.forEach(({ button }, itemIndex) => button.setAttribute("aria-expanded", String(itemIndex === index)));
            const abstract = document.getElementById("publication-abstract");
            abstract.hidden = !node.abstract;
            abstract.open = false;
            document.getElementById("publication-detail-abstract").textContent = node.abstract;
            resourcesPanel.replaceChildren();
            helpers.resources(node.record).forEach(({ label, url }) => {
                const link = document.createElement("a");
                link.className = "btn btn-light";
                link.textContent = label;
                link.href = url;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                resourcesPanel.append(link);
            });
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
            detailBody.scrollTop = 0;
            writeLocation();
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
            const searchTerms = helpers.normalize(query).split(" ").filter(Boolean);
            matchingPublicationIndices = nodes
                .map((node, index) => ({ node, index, score: helpers.searchScore(node.searchIndex, query) }))
                .filter(({ score }) => score >= 0)
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
            renderList();
            emptyResults.hidden = !graphIsActive() || matchingPublicationIndices.length > 0;
            graphContainer.classList.toggle("empty-results", matchingPublicationIndices.length === 0);
            resultsPanel.hidden = currentView !== "list" && (!graphIsActive() || matchingPublicationIndices.length > 0);
            hideTooltip();

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
            return `${node.title} (${node.pub_year}). Topic: ${node.cluster_label}.`;
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
            const viewportBounds = mapViewport.getBoundingClientRect();
            const targetBounds = target.getBoundingClientRect();
            // Keep arrow-key navigation on the scrollable canvas visible.
            const delta = (start, end, lower, upper) => start < lower ? start - lower : end > upper ? end - upper : 0;
            mapViewport.scrollBy({
                left: delta(targetBounds.left, targetBounds.right, viewportBounds.left + 8, viewportBounds.right - 8),
                top: delta(targetBounds.top, targetBounds.bottom, viewportBounds.top + 8, viewportBounds.bottom - 8),
            });
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
        mapViewport.addEventListener("scroll", hideTooltip);

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
            statusElement.hidden = true;
            applySearch(searchInput.value);
            writeLocation({ replace: true });
        });
        searchInput?.addEventListener("keydown", event => {
            if (event.key !== "Enter") {
                return;
            }

            event.preventDefault();
            if (matchingPublicationIndices.length > 0) {
                const best = [...matchingPublicationIndices].sort((a, b) => (
                    helpers.searchScore(nodes[b].searchIndex, searchInput.value)
                    - helpers.searchScore(nodes[a].searchIndex, searchInput.value)
                    || nodes[b].pub_year - nodes[a].pub_year
                ))[0];
                setRovingIndex(best, true);
            }
        });
        searchClearButton?.addEventListener("click", () => {
            if (!searchInput) {
                return;
            }
            searchInput.value = "";
            statusElement.hidden = true;
            applySearch("");
            writeLocation({ replace: true });
            searchInput.focus({ preventScroll: true });
        });
        detailCloseButton?.addEventListener("click", () => {
            clearPublicationDetail({ restoreFocus: true });
        });

        graphContainer?.addEventListener("publicationgraph:visibilitychange", () => {
            restorePublicationLocation();
            hideTooltip();
        });

        clusterLabelLayer = svg.append("g")
            .attr("class", "cluster-label-layer")
            .attr("aria-hidden", "true");
        let simulation;
        let renderedDimensions = "";

        function render() {
            hideTooltip();
            updateToolbarGeometry();
            if (!mapViewport.clientWidth || !mapViewport.clientHeight) return;
            // Preserve readable labels and touch targets instead of squeezing the
            // entire collection into a phone-sized plot.
            const width = Math.max(1024, mapViewport.clientWidth);
            const height = Math.max(720, mapViewport.clientHeight);
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
            radiusScale.range([12.1, Math.min(20, Math.max(16, radiusBase / 60))]);
            renderSizeLegend();
            nodes.forEach(node => {
                node.x_orig = xScale(node.x_data);
                node.y_orig = yScale(node.y_data);
                node.r = radiusScale(node.num_citations);
            });

            if (!simulation) {
                simulation = d3.forceSimulation(nodes)
                    .force("x", d3.forceX(node => node.x_orig).strength(0.15))
                    .force("y", d3.forceY(node => node.y_orig).strength(0.15))
                    .force("collide", d3.forceCollide(node => node.r + 1).strength(1).iterations(3))
                    .stop();
            } else {
                simulation
                    .force("x", d3.forceX(node => node.x_orig).strength(0.15))
                    .force("y", d3.forceY(node => node.y_orig).strength(0.15))
                    .force("collide", d3.forceCollide(node => node.r + 1).strength(1).iterations(3));
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
            mergedLabels.style("display", null).select("text").text(label => label.label);
            const placedLabels = [];
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
                const preferredY = Math.max(
                    labelMargin + labelHeight / 2,
                    Math.min(height - labelMargin - labelHeight / 2, label.y)
                );
                // Move nearby labels apart; omit a label if no clear position is
                // available. Every paper still exposes its topic in its details.
                const offsets = [0, -1, 1, -2, 2, -3, 3, -4, 4];
                const candidates = offsets.map(offset => {
                    const y = preferredY + offset * (labelHeight + 8);
                    return { left: clampedX - labelWidth / 2, right: clampedX + labelWidth / 2,
                        top: y - labelHeight / 2, bottom: y + labelHeight / 2, y };
                });
                const position = candidates.find(candidate => (
                    candidate.top >= labelMargin && candidate.bottom <= height - labelMargin
                    && placedLabels.every(other => candidate.right + 6 <= other.left
                        || candidate.left >= other.right + 6 || candidate.bottom + 6 <= other.top
                        || candidate.top >= other.bottom + 6)
                ));
                group.style("display", position ? null : "none");
                if (!position) return;
                placedLabels.push(position);

                group
                    .attr("transform", `translate(${clampedX}, ${position.y})`)
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
        restorePublicationLocation();
        Object.values(viewButtons).forEach(button => { button.disabled = false; });
        const mapObserver = new ResizeObserver(render);
        mapObserver.observe(mapViewport);
        d3.select(window).on("resize.graph", render);
        window.addEventListener("unload", () => {
            d3.select(window).on("resize.graph", null);
            toolbarObserver.disconnect();
            mapObserver.disconnect();
            simulation?.stop();
        });
    }).catch(showGraphError);
})();
