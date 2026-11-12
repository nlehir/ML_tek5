"""
Perform the k-means algorithm (unsupervised learning)
"""

import os

import matplotlib.pyplot as plt
import numpy as np

IMAGES_FOLDER = "images"


def main(nbs_of_iterations: int) -> None:
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

    # clean images
    clean(IMAGES_FOLDER)

    # we dont initialize the centroids completely randomly
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)

    # initialize the centroid positions
    x_centroids = np.random.uniform(x_min, x_max, 3)
    y_centroids = np.random.uniform(y_min, y_max, 3)
    print("initial centroid positions")
    print(x_centroids, y_centroids)

    # randomly assign the centroids
    # here just used to created a datastructure containing the assignments
    centroids_assignments = np.random.randint(0, 3, nb_samples)

    for iteration in range(nbs_of_iterations):
        print(f"\niteration: {iteration}")
        for datapoint in range(nb_samples):
            """
            Find the closest centroid for this point
            """
            x_point = x[datapoint]
            y_point = y[datapoint]
            distance_0 = (x_point - x_centroids[0]) ** 2 + (
                y_point - y_centroids[0]
            ) ** 2
            distance_1 = (x_point - x_centroids[1]) ** 2 + (
                y_point - y_centroids[1]
            ) ** 2
            distance_2 = (x_point - x_centroids[2]) ** 2 + (
                y_point - y_centroids[2]
            ) ** 2
            distances = [distance_0, distance_1, distance_2]
            # get the index of the closest centroid
            centroid = distances.index(min(distances))
            centroids_assignments[datapoint] = centroid

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
        x_centroids[0] = np.mean(x[cluster_0])
        x_centroids[1] = np.mean(x[cluster_1])
        x_centroids[2] = np.mean(x[cluster_2])
        y_centroids[0] = np.mean(y[cluster_0])
        y_centroids[1] = np.mean(y[cluster_1])
        y_centroids[2] = np.mean(y[cluster_2])
        print("centroids positions")
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


def plot_clustering(
    iteration: int,
    x: np.ndarray,
    y: np.ndarray,
    cluster_0: np.ndarray,
    cluster_1: np.ndarray,
    cluster_2: np.ndarray,
    x_centroids,
    y_centroids,
    step: str,
) -> None:
    """
    Plot the current state of the clustering

    """
    plt.plot(
        x[cluster_0], y[cluster_0], "o", color="darkorange", markersize=3, alpha=0.8
    )
    plt.plot(
        x[cluster_1], y[cluster_1], "o", color="firebrick", markersize=3, alpha=0.8
    )
    plt.plot(
        x[cluster_2], y[cluster_2], "o", color="cornflowerblue", markersize=3, alpha=0.8
    )
    plt.plot(x_centroids, y_centroids, "o", color="lime")
    title = f"update centroids : iteration {iteration} (centroids in green)"
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    figpath = os.path.join(IMAGES_FOLDER, f"it_{iteration}_{step}.pdf")
    plt.savefig(figpath)
    plt.close("all")


def clean(directory):
    for filename in os.listdir(directory):
        os.remove(os.path.join(directory, filename))


if __name__ == "__main__":
    nbs_of_iterations = 10
    main(nbs_of_iterations)
