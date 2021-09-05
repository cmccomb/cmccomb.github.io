// create an array with nodes
var nodes = new vis.DataSet(pubs[0]);

// create an array with edges
var edges = new vis.DataSet(pubs[1]);

// create a network
var container = document.getElementById("graph-container");
var data = {
    nodes: nodes,
    edges: edges,
};

var options = {
    layout: {
        randomSeed: 863946, //Math.floor(Math.random() * 1000000),
    }
}


var network = new vis.Network(container, data, options);

network.stopSimulation();
network.stabilize(1);
console.log(network.getSeed())


network.on("doubleClick", function (params) {
    console.log(params)
    node_id = params.nodes[0];
    paper_name = nodes.get(node_id).title;
    escaped_paper_name = encodeURIComponent(paper_name.replace("Double-click to open ", ""));
    url = "https://scholar.google.com/scholar?hl=en&q=" + escaped_paper_name
    window.open(url, '_blank').focus();
});

