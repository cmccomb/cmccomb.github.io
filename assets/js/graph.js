// create a network
let container = document.getElementById("graph-container");

nodes.push({id:1000, size: 0, shape: "dot"});
nodes.push({id:1001, color: "white", label: "People\nOn Their\nOwn", shape: "box"});
nodes.push({id:1002, color: "white", size: 60, label: "People\nIn\nTeams", shape: "box"});
nodes.push({id:1003, color: "white", size: 60, label: "People\nIn\nOrganizations", shape: "box"});
nodes.push({id:1004, color: "white", size: 60, label: "Design\nApplications", shape: "box"});
nodes.push({id:1005, color: "white", size: 60, label: "Design for\nAdditive\nManufacturing", shape: "box"});
nodes.push({id:1006, color: "white", size: 60, label: "Design\nAlgorithms", shape: "box"});
nodes.push({id:1007, color: "white", size: 60, label: "Human-Computer\nCollaboration", shape: "box"});
nodes.push({id:1008, color: "white", size: 60, label: "Published\nData", shape: "box"});
nodes.push({id:1009, color: "white", size: 60, label: "Design\nDecision\nMaking", shape: "box"});
nodes.push({id:1010, color: "white", size: 60, label: "Engineering\nand Design\nEducation", shape: "box"});

edge_info = [
    [6, 7,24, 26, 29, 32 ,52 ,54, 63, 99, 104, 91, 93, 80, 85, 95],
    [0, 2, 4, 12, 14, 17, 33, 45, 49, 50, 55, 59, 60, 109, 110, 69, 75, 103, 89, 83],
    [20, 107, 78],
    [10, 44, 42, 51, 43, 101, 102, 1],
    [11, 13, 21, 35, 62, 111, 65, 90, 76, 77, 81, 67],
    [3, 15, 18, 22, 23, 19, 36, 39, 48, 64, 58, 56, 112, 106, 94, 86, 84, 96, 97, 100],
    [28, 30, 31,34, 37, 108, 105, 66, 68, 79, 82, 98],
    [8, 9, 16, 27, 38],
    [5, 47, 73, 70],
    [25, 40, 41, 46, 53, 57, 61, 71, 72, 74, 87, 88, 92]
]

edges = []
for (let i = 0; i < edge_info.length; i++) {
    edges.push({
        to: 1000,
        from: i+1001
    })
    for (let j = 0; j < edge_info[i].length; j++) {
        edges.push({
            to: i+1001,
            from: edge_info[i][j]
        })
    }
}


let data = {
    nodes: new vis.DataSet(nodes),
    edges: new vis.DataSet(edges),
};

let options = {
    layout: {
        randomSeed: 863947,
    },
    interaction:{
        tooltipDelay: 10000000,
    },
    nodes:{
        icon: {
            face: 'FontAwesome',
            code: '\uf15b',
            // weight: undefined,
            // size: 50,  //50,
            color:'#212529'
        },
    }
}


let network = new vis.Network(container, data, options);

network.stopSimulation();
network.stabilize(1);


network.on("doubleClick", function (params) {
    let paper_name = data.nodes.get(params.nodes[0]).title;
    let url = "https://scholar.google.com/scholar?hl=en&q=" + encodeURIComponent(paper_name.replace("Double-click to open ", ""));
    window.open(url, '_blank').focus();
});

