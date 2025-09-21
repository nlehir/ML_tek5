"""
    Perform the k-means algorithm (unsupervised learning)
"""
import os

import numpy as np

from utils import plot_clustering, clean

IMAGES_FOLDER = "images"


def cluster_dataset(nbs_of_iterations: int) -> None:
    """
    perform a k-means algorithm
    with 3 clusters on a simple 2D dataset.
    """
    # load the data
    datapath = os.path.join("data", "data_2D.npy")
    data = np.load(datapath)
    x = data[:, 0]
    y = data[:, 1]

    nb_samples = len(x)


    # we dont initialize the centroids completely randomly
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)

    # initialize the centroid positions
    rng = np.random.default_rng()
    x_centroids = rng.uniform(low=x_min, high=x_max, size=3)
    y_centroids = rng.uniform(low=y_min, high=y_max, size=3)
    print("initial centroid positions")
    print(f"x_centroids: {x_centroids}")
    print(f"y_centroids: {y_centroids}")

    # randomly assign the centroids
    # here just used to created a datastructure containing the assignments
    centroids_assignments = rng.integers(low=0, high=3, size=nb_samples)

    for iteration in range(nbs_of_iterations):
        print(f"\niteration: {iteration}")

        """
        Non-vectorized version
        """
        for datapoint in range(nb_samples):
            """
            Find the closest centroid for this point
            add lines here
            """
            pass

        """
        Vectorized version
        """


        """
        Assign each sample to a cluster
        """
        cluster_0 = np.where(centroids_assignments == 0)[0]
        cluster_1 = np.where(centroids_assignments == 1)[0]
        cluster_2 = np.where(centroids_assignments == 2)[0]

        plot_clustering(
            iteration,
            x,
            y,
            cluster_0,
            cluster_1,
            cluster_2,
            x_centroids,
            y_centroids,
            "assign_samples_to_centroid",
        )

        # Update centroids positions
        """
        add lines here
        """
        print(f"x0: {x_centroids[0]:.2f}  y0: {y_centroids[0]:.2f}")
        print(f"x1: {x_centroids[1]:.2f}  y1: {y_centroids[1]:.2f}")
        print(f"x2: {x_centroids[2]:.2f}  y2: {y_centroids[2]:.2f}")

        plot_clustering(
            iteration,
            x,
            y,
            cluster_0,
            cluster_1,
            cluster_2,
            x_centroids,
            y_centroids,
            "move_centroids",
        )


def main() -> None:
    nbs_of_iterations = 10
    clean(IMAGES_FOLDER)
    cluster_dataset(nbs_of_iterations=nbs_of_iterations)


if __name__ == "__main__":
    main()
