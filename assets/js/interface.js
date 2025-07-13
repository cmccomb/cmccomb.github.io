// button stuff
document.getElementById("exit").onclick = function () {
    if (window.matchMedia("(max-width: 768px)").matches) {
        window.open("https://scholar.google.com/citations?user=0P9w_S0AAAAJ&hl=en", "_self");
    } else {
        document.getElementById("profile").remove();
        document.getElementById("footer").remove();
        document.getElementById("graph-container").classList.remove("blur");
        document.getElementById("graph-container").style.pointerEvents = "auto";
    }
}


