(() => {
    "use strict";

    const exploreButton = document.getElementById("exit");
    const graphContainer = document.getElementById("graph-container");
    const graphCloseButton = document.getElementById("graph-close");
    const graphCommandBar = document.getElementById("graph-command-bar");
    const publicationDetail = document.getElementById("publication-detail");
    const publicationDetailClose = document.getElementById("publication-detail-close");
    const publicationSearch = document.getElementById("publication-search");
    const publicationSearchClear = document.getElementById("publication-search-clear");
    const profile = document.getElementById("profile");
    const footer = document.getElementById("footer");
    if (!exploreButton || !graphContainer || !graphCloseButton || !graphCommandBar
        || !publicationDetail || !publicationDetailClose || !publicationSearch
        || !publicationSearchClear || !profile || !footer) return;

    const compactViewport = window.matchMedia("(max-width: 768px)");
    const defaultView = () => compactViewport.matches ? "list" : "map";
    let preferredView = null;
    let profileOpener = exploreButton;
    let browserLoad;
    let recentLoad;

    function updateToolbarGeometry() {
        const bottom = graphCommandBar.getBoundingClientRect().bottom;
        graphContainer.style.setProperty("--browser-content-top", `${bottom + 12}px`);
    }
    new ResizeObserver(updateToolbarGeometry).observe(graphCommandBar);

    function loadScript(src, priority) {
        return new Promise((resolve, reject) => {
            const script = document.createElement("script");
            script.fetchPriority = priority;
            script.src = src;
            script.addEventListener("load", resolve, { once: true });
            script.addEventListener("error", () => {
                script.remove();
                reject(new Error(`Unable to load ${src}`));
            }, { once: true });
            document.head.append(script);
        });
    }

    function showBrowserError(error) {
        console.error("Unable to load the publication browser", error);
        graphContainer.classList.add("data-unavailable");
        publicationSearch.disabled = true;
        publicationSearchClear.disabled = true;
        document.getElementById("publication-search-status").textContent = "Publication search is unavailable.";
        graphContainer.querySelectorAll(".publication-view-controls button").forEach(button => {
            button.disabled = true;
            button.setAttribute("aria-pressed", "false");
        });
        document.getElementById("publication-results").hidden = true;
        document.getElementById("publication-empty").hidden = true;
        document.getElementById("graph-status-message").textContent = "The publication browser is temporarily unavailable.";
        document.getElementById("graph-status-dismiss").hidden = true;
        document.getElementById("graph-status-scholar").hidden = false;
        document.getElementById("graph-status").hidden = false;
    }

    graphContainer.addEventListener("publicationgraph:unavailable", event => showBrowserError(event.detail));
    graphContainer.addEventListener("publicationgraph:ready", () => {
        if (!graphContainer.hidden && document.activeElement === graphCloseButton) {
            publicationSearch.focus({ preventScroll: true });
        }
    });

    function loadBrowser(priority = "auto") {
        if (browserLoad) return;
        const d3Ready = window.d3 ? Promise.resolve() : loadScript(graphContainer.dataset.d3Src, priority);
        browserLoad = d3Ready
            .then(() => loadScript(graphContainer.dataset.browserSrc, priority))
            .catch(error => {
                if (priority === "low") {
                    browserLoad = null;
                    if (!graphContainer.hidden) loadBrowser();
                } else {
                    showBrowserError(error);
                }
            });
    }

    function scheduleBrowserPreload() {
        function preload() {
            const connection = navigator.connection;
            if (browserLoad || !graphContainer.hidden || connection?.saveData
                || ["slow-2g", "2g", "3g"].includes(connection?.effectiveType)) return;
            if (document.hidden) {
                document.addEventListener("visibilitychange", preload, { once: true });
                return;
            }
            loadBrowser("low");
        }

        const afterLoad = () => recentLoad.then(() => {
            window.setTimeout(() => {
                if (window.requestIdleCallback) window.requestIdleCallback(preload);
                else preload();
            }, 2000);
        });
        if (document.readyState === "complete") afterLoad();
        else window.addEventListener("load", afterLoad, { once: true });
    }

    function loadRecentPublications() {
        const panel = document.getElementById("recent-publications");
        recentLoad ??= fetch(panel.dataset.src)
            .then(response => {
                if (!response.ok) throw new Error(`Recent publications returned HTTP ${response.status}`);
                return response.json();
            })
            .then(records => renderRecentPublications(panel, records))
            .catch(error => console.warn("Unable to load recent publications", error));
    }

    function renderRecentPublications(panel, records) {
        const helpers = window.PublicationHelpers;
        const list = panel.querySelector(".recent-publications-list");
        for (const record of helpers.sampleRecentPublications(records, new Date().getFullYear())) {
            const item = document.createElement("li");
            const meta = document.createElement("div");
            meta.className = "recent-publication-meta";
            const names = helpers.authors(record.bib_dict.author, { compact: true });
            meta.textContent = names ? `${names} (${record.pub_year}).` : String(record.pub_year);

            const title = document.createElement("a");
            title.className = "recent-publication-title";
            const params = new URLSearchParams({ view: "list", paper: record.author_pub_id });
            title.href = `?${params}`;
            title.textContent = record.bib_dict.title;
            item.append(meta, title);

            const venue = helpers.venue(record);
            if (venue) {
                const source = document.createElement("div");
                source.className = "recent-publication-meta";
                source.textContent = venue;
                item.append(source);
            }
            list.append(item);
        }
        if (!list.childElementCount) return;
        const restoreFocus = document.activeElement === exploreButton;
        list.after(exploreButton);
        panel.hidden = false;
        if (restoreFocus) exploreButton.focus();
    }

    function updateView(view) {
        graphContainer.dataset.publicationView = view;
        exploreButton.href = `?view=${view}`;
    }

    function setRegionVisibility(element, isVisible) {
        element.hidden = !isVisible;
        element.setAttribute("aria-hidden", String(!isVisible));
    }

    function restoreLocation({ focus = false } = {}) {
        const params = new URLSearchParams(window.location.search);
        const isVisible = params.has("view") || params.has("paper") || params.has("q");
        const requestedView = params.get("view");
        if (requestedView === "list" || requestedView === "map") preferredView = requestedView;
        updateView(preferredView || defaultView());
        setRegionVisibility(profile, !isVisible);
        setRegionVisibility(footer, !isVisible);
        graphContainer.inert = !isVisible;
        setRegionVisibility(graphContainer, isVisible);
        graphContainer.classList.toggle("graph-active", isVisible);
        graphCloseButton.hidden = !isVisible;
        setRegionVisibility(graphCommandBar, isVisible);
        exploreButton.setAttribute("aria-expanded", String(isVisible));
        updateToolbarGeometry();
        if (isVisible) loadBrowser();
        else loadRecentPublications();
        graphContainer.dispatchEvent(new CustomEvent("publicationgraph:visibilitychange", {
            detail: { isVisible },
        }));
        if (focus) {
            (isVisible ? (publicationSearch.disabled ? graphCloseButton : publicationSearch)
                : profileOpener).focus({ preventScroll: isVisible });
        }
    }

    profile.addEventListener("click", event => {
        const paperLink = event.target.closest(".recent-publication-title");
        if (!paperLink && !exploreButton.contains(event.target)) return;
        if (event.button !== 0 || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
        event.preventDefault();
        profileOpener = paperLink || exploreButton;
        const url = new URL(paperLink?.href || window.location.href);
        if (!paperLink) url.searchParams.set("view", preferredView || defaultView());
        window.history.pushState(null, "", url);
        // When opening a paper, the publication browser focuses its detail title.
        restoreLocation({ focus: !paperLink });
    });
    graphCloseButton.addEventListener("click", () => {
        const url = new URL(window.location.href);
        for (const key of ["view", "q", "paper"]) url.searchParams.delete(key);
        window.history.pushState(null, "", url);
        restoreLocation({ focus: true });
    });
    graphContainer.addEventListener("keydown", event => {
        if (event.key !== "Escape" || !graphContainer.classList.contains("graph-active")) return;
        event.preventDefault();
        if (!publicationDetail.hidden) publicationDetailClose.click();
        else if (publicationSearch.value) publicationSearchClear.click();
        else graphCloseButton.click();
    });
    window.addEventListener("popstate", () => restoreLocation({ focus: true }));
    graphContainer.addEventListener("publicationgraph:viewchange", event => {
        preferredView = event.detail.view;
        updateView(preferredView);
    });
    compactViewport.addEventListener("change", () => {
        if (!preferredView && !graphContainer.classList.contains("graph-active")) restoreLocation();
    });
    restoreLocation();
    if (graphContainer.hidden) scheduleBrowserPreload();
})();
