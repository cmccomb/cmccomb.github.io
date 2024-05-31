// button stuff
document.getElementById("exit").onclick = function () {
    if (document.getElementById("graph-container").style.visibility == "visible") {
        document.getElementById("profile").remove();
        document.getElementById("footer").remove();
        document.getElementById("graph-container").classList.remove("blur");
        document.getElementById("graph-container").style.pointerEvents = "auto";
    } else {
        window.open("https://ccm-search-my-publications.hf.space/", "_self");
    }
}


