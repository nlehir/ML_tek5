"""
Study the normalized cu heursitic for obtaining a relevant number of clusters
in a clustering situation.

The clustering is performed by a Spectral Clustering.

Spectral Clustering works with an adjacency matrix
or a similarity matrix.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering

# load the data
adjacency_matrix = np.load(os.path.join("data", "adjacency_matrix.npy"))


def cluster_and_compute_normalized_cut(
    nb_clusters: int,
    adjacency_matrix: np.ndarray,
) -> float:
    n = adjacency_matrix.shape[0]
    """
    EDIT
    """
    normalized_cut = 1
    return normalized_cut


def main() -> None:
    normalized_cuts = list()
    max_nb_clusters = 10
    tried_nb_clusters = range(1, max_nb_clusters)

    for nb_clusters in tried_nb_clusters:
        print(f"======\nnb clusters: {nb_clusters}")
        normalized_cuts.append(
            cluster_and_compute_normalized_cut(
                nb_clusters=nb_clusters,
                adjacency_matrix=adjacency_matrix,
            )
        )

    plt.plot(tried_nb_clusters, normalized_cuts, "o")
    plt.title("normalized cut heuristic")
    plt.xlabel("nb clusters")
    plt.ylabel("normalized cut")
    plt.savefig(os.path.join("images", "normalized_cuts.pdf"))


if __name__ == "__main__":
    main()
