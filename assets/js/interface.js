// button stuff
document.getElementById("exit").onclick = function () {
    if (document.documentElement.clientWidth > 768) {
        document.getElementById("profile").remove();
        document.getElementById("footer").remove();
        document.getElementById("graph-container").classList.remove("blur");
        document.getElementById("graph-container").style.pointerEvents = "auto";
    } else {
        window.open("https://ccm-chat-with-publications.hf.space/", "_self");
    }
}


