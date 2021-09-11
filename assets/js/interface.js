
// button stuff
document.getElementById("exit").onclick= function() {
    document.getElementById("profile").remove();
    document.getElementById("footer").remove();
    document.getElementById("graph-container").classList.remove("blur");
    network.setOptions({
        interaction:{
            tooltipDelay: 0,
        }
    });

    network.fit({animation: {duration: 2000}});
}
