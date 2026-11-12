"""
Study the normalized cu heursitic for obtaining a relevant number of clusters
in a clustering situation.

The clustering is performed by a Spectral Clustering.

Spectral Clustering works with an adjacency matrix
or a similarity matrix.
"""

import os
from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt

# load the data
adjacency_matrix = np.load(os.path.join("data", "adjacency_matrix.npy"))


def cluster_and_compute_normalized_cut(
    nb_clusters: int,
    adjacency_matrix: np.ndarray,
) -> float:
    n = adjacency_matrix.shape[0]
    sc = SpectralClustering(nb_clusters, affinity="precomputed")
    sc.fit_predict(adjacency_matrix)

    # compute the normalized cut
    normalized_cut = 0
    for cluster_index in range(nb_clusters):
        cluster = np.where(sc.labels_ == cluster_index)[0]

        # compute degree of the cluster
        cluster_adjacencies = adjacency_matrix[cluster]
        cluster_degree = cluster_adjacencies.sum()

        # compute the number of outside connections
        complementary = np.setdiff1d(range(n), cluster)
        cluster_cut = adjacency_matrix[cluster][:, complementary].sum()

        # apply definition of normalized cut
        normalized_cut += cluster_cut / cluster_degree

    print(f"normalized cut: {normalized_cut}")
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
