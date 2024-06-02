import textwrap

import bs4
import datasets
import numpy
import plotly.express
import plotly.graph_objects
import sklearn.decomposition
import sklearn.manifold

# General settings
N = 10  # number of top citations to show
padding = 1.2  # Padding to give around teh annotation

# Load data from huggingface
citations = datasets.load_dataset("ccm/publications")["train"].to_pandas()

# Make a t-SNE embedding of the embeddings
tsne_embeddings = sklearn.manifold.TSNE(n_components=2, random_state=42).fit_transform(
    numpy.stack(citations["embedding"].values)
)

# Next do a PCA in order to figure out how best to orient the plot
oriented_tsne_embeddings = sklearn.decomposition.PCA(
    n_components=2, random_state=42
).fit_transform(tsne_embeddings)
citations["y"] = oriented_tsne_embeddings[:, 0]
citations["x"] = oriented_tsne_embeddings[:, 1]

# Add a score which will be used to weight the point in teh plot
citations["score"] = [numpy.log(n + 1) for n in citations["num_citations"]]
citations.sort_values("score", ascending=True, inplace=True)

# Create a formatted title for each of hte publications
citations["TitleFormatted"] = [
    textwrap.fill(pub["title"], 40).replace("\n", "<br>")
    for pub in citations["bib_dict"]
]

# Plot and create tooltips for each publication
fig = plotly.express.scatter(
    citations, x="x", y="y", size="score", custom_data=["TitleFormatted"]
)
fig.update_traces(
    hovertemplate="%{customdata[0]}",
    marker=dict(color="#73C1D4", opacity=1),
)

# Now let's just look at the top N publications
top = citations.tail(N)

# Let's get a right group and a left group to help with plotting
top.sort_values("x", inplace=True)
left = top.head(int(N / 2)).sort_values("y", ascending=False)
right = top.tail(int(N / 2)).sort_values("y", ascending=False)

# Figure out the upper and lower extents of the data
upper_bound = numpy.max(citations["y"].values)
lower_bound = numpy.min(citations["y"].values)

# Create data packages fro the left and rigth sides
left_data = (numpy.min(citations["x"].values), left, "right")
right_data = (numpy.max(citations["x"].values), right, "left")

# Add annotations to the plot
for extent, data, alignment in [left_data, right_data]:
    for i in range(int(N / 2)):

        # Create an extended citation
        extended_citation = (
            data["bib_dict"].values[i]["author"].split(" and ")[0].split(" ")[-1]
            + " et al. ("
            + str(int(data["bib_dict"].values[i]["pub_year"]))
            + ') "'
            + data["bib_dict"].values[i]["title"]
            + '."'
        )

        # Create full citation with a link
        formatted_citation_with_link = (
            textwrap.fill(extended_citation, 40).replace("\n", "<br>")
            + '<br><a style="color:white" href="'
            + str(data["pub_url"].values[i])
            + '">>>> Read Paper </a>'
        )

        # Add initial annotation with any label
        fig.add_annotation(
            x=data["x"].values[i],
            ax=extent,
            axref="x",
            y=data["y"].values[i],
            ay=data["y"].values[i],
            ayref="y",
            xanchor=alignment,
            yanchor="middle",
            arrowcolor="#73C1D4",
        )

        # Add final annotation with label
        fig.add_annotation(
            align=alignment,
            text=formatted_citation_with_link,
            x=extent,
            ax=padding * extent,
            axref="x",
            y=data["y"].values[i],
            ay=(upper_bound - lower_bound) * (int(N / 2) - i - 0.5) / int(N / 2)
            + lower_bound,
            ayref="y",
            arrowcolor="#73C1D4",
            xanchor=alignment,
            yanchor="middle",
            font=dict(size=10, color="#73C1D4"),
        )

# Make things pretty - remove axes
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

# Make things pretty - set a good margin and good colors
fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor="#191C1F",
    hoverlabel=dict(
        bgcolor="white",
    ),
)

# Make things pretty - remove the plotly logo
fig.show(
    config={
        "displaylogo": False,
    }
)


# Export HTML and PNG
fig.write_html("pubs.html")

html = open("pubs.html")
soup = bs4.BeautifulSoup(html)
open("_includes/graph.html", "w").write(
    "".join([str(x) for x in soup.html.body.div.contents])
)
