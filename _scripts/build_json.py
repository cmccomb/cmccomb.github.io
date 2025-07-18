"""
This script builds a JSON file that contains the t-SNE embeddings of the publications. Specifically, it
1. Loads the dataset from Hugging Face.
2. Computes t-SNE embeddings for the publication embeddings.
3. Applies PCA to orient the t-SNE embeddings.
4. Saves the resulting data to a JSON file.
"""

import json
import os

import datasets
import numpy
import sklearn.decomposition
import sklearn.manifold

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

# Create a payload for the JSON file
payload = citations[
    [
        "x",
        "y",
        "author_pub_id",
        "pub_year",
        "num_citations",
        "bib_dict",
    ]
].to_dict(orient="records")

# Make directory
os.makedirs(
    os.path.join(os.environ.get("GITHUB_WORKSPACE", ".."), "assets/json"), exist_ok=True
)

# Save to directory
with open(
    os.path.join(os.environ.get("GITHUB_WORKSPACE", ".."), "assets/json/pubs.json"), "w"
) as f:
    json.dump(payload, f, indent=2)
