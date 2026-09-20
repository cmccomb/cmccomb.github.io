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
        graphContainer.setAttribute("aria-hidden", String(!isVisible));
        graphContainer.classList.toggle("blur", !isVisible);
        graphContainer.classList.toggle("graph-active", isVisible);
        graphCloseButton.hidden = !isVisible;
        setRegionVisibility(graphCommandBar, isVisible);
        exploreButton.setAttribute("aria-expanded", String(isVisible));
        graphContainer.dispatchEvent(new CustomEvent("publicationgraph:visibilitychange", {
            detail: { isVisible },
        }));
        if (focus) {
            (isVisible ? (publicationSearch.disabled ? graphCloseButton : publicationSearch)
                : exploreButton).focus({ preventScroll: true });
        }
    }

    exploreButton.addEventListener("click", event => {
        if (event.button !== 0 || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
        event.preventDefault();
        const url = new URL(window.location.href);
        url.searchParams.set("view", preferredView || defaultView());
        window.history.pushState(null, "", url);
        restoreLocation({ focus: true });
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
})();
