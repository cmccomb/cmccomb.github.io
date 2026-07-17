(() => {
    "use strict";

    const exploreButton = document.getElementById("exit");
    if (!exploreButton) {
        return;
    }

    exploreButton.addEventListener("click", () => {
        if (window.matchMedia("(max-width: 768px)").matches) {
            window.location.assign("https://scholar.google.com/citations?user=0P9w_S0AAAAJ&hl=en");
            return;
        }

        document.getElementById("profile")?.remove();
        document.getElementById("footer")?.remove();
        const graphContainer = document.getElementById("graph-container");
        graphContainer?.classList.remove("blur");
        graphContainer?.classList.add("graph-active");
    });
})();
