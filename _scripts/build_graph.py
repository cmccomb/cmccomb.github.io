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
halfN = int(N / 2)  # half of the number of top citations to show
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
citations["x"] = oriented_tsne_embeddings[:, 0]
citations["y"] = oriented_tsne_embeddings[:, 1]

# Add a score which will be used to weight the point in teh plot
citations["score"] = [numpy.log(n + 1) for n in citations["num_citations"]]
citations.sort_values("score", ascending=True, inplace=True)

# Create a formatted title for each of hte publications
citations["TitleFormatted"] = [
    textwrap.fill(pub["title"], 40).replace("\n", "<br>")
    for pub in citations["bib_dict"]
]

# Boost top 10 scores by 1.5
citations.loc[citations["score"].nlargest(N).index, "score"] *= 1.5

# Plot and create tooltips for each publication
fig = plotly.express.scatter(
    citations, x="x", y="y", size="score", custom_data=["TitleFormatted"]
)
fig.update_traces(
    hovertemplate="%{customdata[0]}",
    marker=dict(color="#73C1D4", opacity=1),
)

citations.sort_values("score", ascending=False, inplace=True)

centroid = citations[["x", "y"]].mean()

for i in range(N):
    # Add final annotation with label
    fig.add_annotation(
        text="<b>#" + str(int(i + 1)) + "</b>",
        x=citations["x"].values[i],
        y=citations["y"].values[i],
        font=dict(size=10, color="white"),
        xclick=citations["x"].values[i],
        yclick=citations["y"].values[i],
        standoff=None,
        showarrow=False,
        axref="x",
        ayref="y",
    )

# Add annotations to the plot
for i in range(citations.shape[0]):

    # Create an extended citation
    extended_citation = (
        citations["bib_dict"].values[i]["author"].split(" and ")[0].split(" ")[-1]
        + " et al. ("
        + str(int(citations["bib_dict"].values[i]["pub_year"] or 2024))
        + ') "'
        + citations["bib_dict"].values[i]["title"]
        + '."'
    )

    # Create full citation with a link
    formatted_citation_with_link = (
        textwrap.fill(extended_citation, 40).replace("\n", "<br>")
        + '<br><a style="color:white" href="'
        + "https://scholar.google.com/citations?view_op=view_citation&citation_for_view="
        + citations["author_pub_id"].values[i]
        + '">>>> Read Paper </a>'
    )

    # angle from centroid is
    angle = numpy.arctan2(
        citations["y"].values[i] - centroid[1], citations["x"].values[i] - centroid[0]
    )

    # Displace notation location in direction away from centroid
    x_displacement = padding * numpy.cos(angle)
    y_displacement = padding * numpy.sin(angle)

    # Set x and y anchor alignemnt based on angle
    x_anchor = "right" if x_displacement < 0 else "left"
    y_anchor = "top" if y_displacement < 0 else "bottom"

    # Add final annotation with label
    fig.add_annotation(
        text=formatted_citation_with_link,
        arrowcolor="#73C1D4",
        font=dict(size=10, color="#73C1D4"),
        clicktoshow="onoff",
        align="left",
        xanchor=x_anchor,
        yanchor=y_anchor,
        bgcolor="#191C1F",
        bordercolor="#73C1D4",
        x=citations["x"].values[i],
        y=citations["y"].values[i],
        ax=citations["x"].values[i] + x_displacement,
        ay=citations["y"].values[i] + y_displacement,
        yclick=citations["y"].values[i],
        xclick=citations["x"].values[i],
        axref="x",
        ayref="y",
        visible=True if i < 10 else False,
        # Standoff as far as markersize requires adaptively
        standoff=15,
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
