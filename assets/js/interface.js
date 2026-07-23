(() => {
    "use strict";

    const exploreButton = document.getElementById("exit");
    const graphContainer = document.getElementById("graph-container");
    const graphCloseButton = document.getElementById("graph-close");
    const profile = document.getElementById("profile");
    const footer = document.getElementById("footer");
    const mobileQuery = window.matchMedia("(max-width: 768px)");

    if (
        !exploreButton
        || !graphContainer
        || !graphCloseButton
        || !profile
        || !footer
    ) {
        return;
    }

    function setRegionVisibility(element, isVisible) {
        element.hidden = !isVisible;
        element.setAttribute("aria-hidden", String(!isVisible));
    }

    function announceGraphVisibility(isVisible) {
        graphContainer.dispatchEvent(new CustomEvent("publicationgraph:visibilitychange", {
            detail: { isVisible },
        }));
    }

    function openGraph(event) {
        const isPlainPrimaryClick = (
            event.button === 0
            && !event.altKey
            && !event.ctrlKey
            && !event.metaKey
            && !event.shiftKey
        );

        if (mobileQuery.matches || !isPlainPrimaryClick) {
            return;
        }

        event.preventDefault();
        setRegionVisibility(profile, false);
        setRegionVisibility(footer, false);
        graphContainer.inert = false;
        graphContainer.setAttribute("aria-hidden", "false");
        graphContainer.classList.remove("blur");
        graphContainer.classList.add("graph-active");
        graphCloseButton.hidden = false;
        exploreButton.setAttribute("aria-expanded", "true");
        announceGraphVisibility(true);
        graphCloseButton.focus({ preventScroll: true });
    }

    function closeGraph() {
        graphCloseButton.hidden = true;
        graphContainer.classList.remove("graph-active");
        graphContainer.classList.add("blur");
        graphContainer.setAttribute("aria-hidden", "true");
        graphContainer.inert = true;
        setRegionVisibility(profile, true);
        setRegionVisibility(footer, true);
        exploreButton.setAttribute("aria-expanded", "false");
        announceGraphVisibility(false);
        exploreButton.focus({ preventScroll: true });
    }

    exploreButton.addEventListener("click", openGraph);
    graphCloseButton.addEventListener("click", closeGraph);
    graphContainer.addEventListener("keydown", event => {
        if (event.key === "Escape" && graphContainer.classList.contains("graph-active")) {
            event.preventDefault();
            closeGraph();
        }
    });
})();
