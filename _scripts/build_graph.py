import textwrap

import bs4
import datasets
import numpy
import plotly.express
import plotly.graph_objects
import sklearn.decomposition
import sklearn.manifold

# General settings
padding = 1.2  # Padding to give around teh annotation

# Load data from huggingface
citations = datasets.load_dataset("ccm/publications")["train"].to_pandas()

# Add a column for the year as a number, based on the bib_dict
citations["pub_year"] = [
    int(pub["pub_year"]) if pub["pub_year"] is not None else 2025
    for pub in citations["bib_dict"]
]

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
citations["score"] = [numpy.log(n + 1) + 1 for n in citations["num_citations"]]
citations.sort_values("score", ascending=True, inplace=True)

# Create a formatted title for each of hte publications
citations["TitleFormatted"] = [
    textwrap.fill(pub["title"], 40).replace("\n", "<br>")
    for pub in citations["bib_dict"]
]

# Plot and create tooltips for each publication
fig = plotly.express.scatter(
    citations,
    x="x",
    y="y",
    size="score",
    color="pub_year",
    custom_data=["TitleFormatted"],
    color_continuous_scale=plotly.colors.sequential.Agsunset,
)

fig.update_traces(hovertemplate="%{customdata[0]}")
fig.update_coloraxes(showscale=False)

centroid = citations[["x", "y"]].mean()

# Add annotations to the plot
for i in range(citations.shape[0]):

    # Create an extended citation
    extended_citation = (
        citations["bib_dict"].values[i]["author"].split(" and ")[0].split(" ")[-1]
        + " et al. ("
        + str(int(citations["bib_dict"].values[i]["pub_year"] or 2025))
        + ') "'
        + citations["bib_dict"].values[i]["title"]
        + '."'
    )

    # Create full citation with a link
    formatted_citation_with_link = (
        textwrap.fill(extended_citation, 40).replace("\n", "<br>")
        + '<br><a style="color: PaleVioletRed; text-decoration: underline;" href="'
        + "https://scholar.google.com/citations?view_op=view_citation&citation_for_view="
        + citations["author_pub_id"].values[i]
        + '">Read Paper 🔗</a>'
    )

    # Displace notation location in direction away from centroid
    x_displacement = padding
    y_displacement = 0

    # Set x and y anchor alignment based on angle
    x_anchor = "right"
    y_anchor = "middle"

    # Add final annotation with label
    fig.add_annotation(
        text=formatted_citation_with_link,
        arrowcolor="#444444",
        font=dict(size=12, color="#444444"),
        clicktoshow="onoff",
        align="left",
        xanchor=x_anchor,
        yanchor=y_anchor,
        bgcolor="#FFFFFF",
        bordercolor="#444444",
        x=citations["x"].values[i],
        y=citations["y"].values[i],
        ax=citations["x"].values[i] - x_displacement,
        ay=citations["y"].values[i] + y_displacement,
        yclick=citations["y"].values[i],
        xclick=citations["x"].values[i],
        axref="x",
        ayref="y",
        visible=False,
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
    "{% raw %}"
    + "".join([str(x) for x in soup.html.body.div.contents])
    + "{% endraw %}"
)
