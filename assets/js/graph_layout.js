(() => {
    "use strict";

    const svg = d3.select("#publication-graph");
    const tooltip = d3.select(".tooltip");
    const statusElement = document.getElementById("graph-status");
    const statusMessage = document.getElementById("graph-status-message");
    const statusDismiss = document.getElementById("graph-status-dismiss");
    const graphContainer = document.getElementById("graph-container");
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

    statusDismiss.addEventListener("click", () => {
        statusElement.hidden = true;
        searchInput.focus({ preventScroll: true });
    });

    function asFiniteNumber(value) {
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    }

    const publicationURL = new URL("assets/json/pubs.json", document.baseURI);
    publicationURL.searchParams.set("v", graphContainer.dataset.siteVersion);
    const browserScript = new URL(document.currentScript.src);
    const mapScript = new URL("publication-map.js", browserScript);
    mapScript.search = browserScript.search;
    Promise.all([d3.json(publicationURL.toString()), import(mapScript.href)]).then(([rawPayload, { createPublicationMap }]) => {
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

        const yearExtent = d3.extent(records, record => record.pub_year);
        const colorScale = d3.scaleSequential(d3.interpolateMagma).domain(yearExtent);

        const nodes = records.map((record, index) => {
            const clusterSummary = clusterSummaryById.get(String(record.cluster_id));
            const clusterLabel = typeof clusterSummary?.label === "string" && clusterSummary.label
                ? helpers.displayClusterLabel(clusterSummary.label)
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
                ], title),
            };
        });

        const map = createPublicationMap({ svg, graphContainer, mapViewport, nodes, colorScale, hideTooltip });
        const { publicationControls } = map;
        const publicationControlNodes = publicationControls.nodes();
        let matchingPublicationIndices = nodes.map((_node, index) => index);
        let activePublicationIndex = 0;
        let selectedPublicationIndex = null;
        let currentView = graphContainer.dataset.publicationView;
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
            authors.textContent = helpers.authors(node.author);
            const meta = document.createElement("span");
            meta.className = "publication-result-meta";
            meta.textContent = `${node.pub_year} · Map neighborhood: ${node.cluster_label} · ${node.num_citations} citation${node.num_citations === 1 ? "" : "s"}`;
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
            if (layoutChanged) map.render();
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
            setView(graphContainer.dataset.publicationView, { save: false });
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
                detailMeta.textContent = `${node.pub_year} · ${node.num_citations} citation${node.num_citations === 1 ? "" : "s"} · Map neighborhood: ${node.cluster_label}`;
            }
            if (detailCitation) {
                detailCitation.textContent = [
                    helpers.authors(node.author),
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
            const visibleClusterIds = new Set(
                matchingPublicationIndices.map(index => String(nodes[index].cluster_id))
            );
            const hasQuery = Boolean(searchInput?.value.trim());
            map.setHiddenLabels(label => hasQuery && !visibleClusterIds.has(String(label.clusterId)));
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
            return `${node.title} (${node.pub_year}).`;
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
            const target = event.currentTarget.querySelector(".publication-hit-target")
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

        searchInput.disabled = false;
        restorePublicationLocation();
        map.render();
        Object.values(viewButtons).forEach(button => { button.disabled = false; });
        graphContainer.dispatchEvent(new Event("publicationgraph:ready"));
    }).catch(error => {
        graphContainer.dispatchEvent(new CustomEvent("publicationgraph:unavailable", { detail: error }));
    });
})();
