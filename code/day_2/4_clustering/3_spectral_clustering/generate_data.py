"""
Generate data to illustrate the normalized cut heuristic.

The data consist in three 2D clusters, with tunable standard deviations.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


def main() -> None:
    std_1 = 1 * 0.1
    std_2 = 1 * 0.2
    std_3 = 1 * 1

    cluster_1 = np.random.normal((1, 4), std_1, (10, 2))
    cluster_2 = np.random.normal((2, 2), std_2, (20, 2))
    cluster_3 = np.random.normal((-2, -2), std_3, (30, 2))

    data = np.concatenate((cluster_1, cluster_2))
    data = np.concatenate((data, cluster_3))
    np.random.shuffle(data)

    nb_points = data.shape[0]
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y, "o")
    plt.title("data to cluster")
    plt.savefig(os.path.join("images", "data_to_cluster.pdf"))
    plt.close()

    # build a matrix of distances
    distances = cdist(XA=data, XB=data)
    # we use the standard deviation to compute the similarity
    similarity = np.exp(-distances / distances.std())
    print(f"distances standard deviation: {distances.std()}")
    print(similarity)
    plt.imshow(similarity)
    plt.savefig(os.path.join("images", "similarity.pdf"))
    plt.close()

    # threshold = distances.std()
    selection = np.where(similarity > 0.5)
    adjacency_matrix = np.zeros(distances.shape)
    adjacency_matrix[selection] = 1
    plt.imshow(adjacency_matrix)
    plt.savefig(os.path.join("images", "adjacency_matrix.pdf"))
    plt.close()

    # save the data
    np.save(os.path.join("data", "similarity"), similarity)
    np.save(os.path.join("data", "adjacency_matrix"), adjacency_matrix)
    np.save(os.path.join("data", "distances"), distances)


if __name__ == "__main__":
    main()
