// button stuff
document.getElementById("exit").onclick = function () {
    document.getElementById("profile").remove();
    document.getElementById("footer").remove();
    document.getElementById("graph-container").classList.remove("blur");
    // document.getElementById("graph-container").classList.remove("unselectable");
    document.getElementById("graph-container").style.visibility = "visible";
    document.getElementById("graph-container").style.pointerEvents = "auto";
    network.setOptions({
        interaction: {
            tooltipDelay: 0,
        }
    });

    network.fit({animation: {duration: 2000}});
}


