import plotly.express
import plotly.graph_objects
import sklearn.manifold
import sklearn.decomposition
import textwrap
import numpy
import datasets
import bs4

citations = datasets.load_dataset("ccm/publications")["train"].to_pandas()

tsne_embeddings = sklearn.manifold.TSNE(n_components=2, random_state=42).fit_transform(numpy.stack(citations["embedding"].values))
tsne_embeddings = sklearn.decomposition.PCA(n_components=2, random_state=42).fit_transform(tsne_embeddings)
citations["y"] = tsne_embeddings[:,0]
citations["x"] = tsne_embeddings[:,1]

citations["score"] = [numpy.log(n+1) for n in citations["num_citations"]]

citations.sort_values("score", ascending=True, inplace=True)

# Visualize

citations["TitleFormatted"] = [textwrap.fill(pub["title"], 40).replace("\n", "<br>") for pub in citations["bib_dict"]]

N = 10
padding = 1.2

top = citations.tail(N)

fig = plotly.express.scatter(citations, x="x", y="y", size="score", custom_data=["TitleFormatted"])

fig.update_traces(
    hovertemplate="%{customdata[0]}",
    marker=dict(color="#73C1D4", opacity=1),
    )

top.sort_values("x", inplace=True)
left = top.head(int(N/2)).sort_values("y", ascending=False)
right = top.tail(int(N/2)).sort_values("y", ascending=False)


fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)


top_extent = numpy.max(citations["y"].values)
bottom_extent = numpy.min(citations["y"].values)

for (extent, data, alignment) in [(numpy.min(citations["x"].values), left, "right"), (numpy.max(citations["x"].values), right, "left")]:
    for i in range(int(N/2)):

        formatted_citation = data["bib_dict"].values[i]["author"].split(" and ")[0].split(" ")[-1] + " et al. (" + str(int(data["bib_dict"].values[i]["pub_year"])) + ") \"" +data["bib_dict"].values[i]["title"] + ".\""
        formatted_citation_with_link = textwrap.fill(formatted_citation, 40).replace("\n", "<br>") + "<br><a style=\"color:white\" href=\"" + data["pub_url"].values[i] + "\">>>> Read Paper </a>"    


        fig.add_annotation(
            x = data["x"].values[i],
            ax = extent,
            axref = "x",
            y = data["y"].values[i],
            ay = data["y"].values[i],
            ayref = "y",
            xanchor = alignment,
            yanchor = 'middle',
            arrowcolor = "#73C1D4",
        )

        fig.add_annotation(
            align = alignment,
            text = formatted_citation_with_link ,
            x = extent,
            ax = padding*extent,
            axref = "x",
            y = data["y"].values[i],
            ay = (top_extent-bottom_extent)*(int(N/2) - i - 0.5)/int(N/2) + bottom_extent,
            ayref = "y",
            arrowcolor = "#73C1D4",
            xanchor = alignment,
            yanchor = 'middle',
            font = dict(
                size = 10,
                color = "#73C1D4"
            ),
        )



fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor="#191C1F",
    hoverlabel=dict(
        bgcolor="white",
    )
)

fig.show(config={
    'displaylogo': False,
    # 'modeBarButtonsToRemove': ['select', 'lasso2d']
    }
)


# Export HTML and PNG

fig.write_html("pubs.html")

from bs4 import BeautifulSoup

html = open("pubs.html")

soup = BeautifulSoup(html)

open("_includes/graph.html", "w").write("".join([str(x) for x in soup.html.body.div.contents]))
