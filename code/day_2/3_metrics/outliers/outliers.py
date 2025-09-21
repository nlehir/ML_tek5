"""
    Z score
"""

import matplotlib.pyplot as plt
import numpy as np


def compute_outliers(data, metric, factor):
    center = np.mean(data, axis=0)
    if metric == "euclidean":
        metric = "euclidean"
        differences = data - center
        distance_to_center = np.linalg.norm(differences, axis=1)
    elif metric == "custom":
        std = data.std(axis=0)
        differences = data - center
        differences /= std
        distance_to_center = np.linalg.norm(differences, ord=1, axis=1)
    else:
        raise ValueError("Unknown metric")

    mean_distance_to_center = distance_to_center.mean()
    points_considered_outliers = data[distance_to_center > factor*mean_distance_to_center]

    return points_considered_outliers


def plot_data_and_outliers(data, metric, factor):
    plt.plot(data[:, 0], data[:, 1], "o", markersize=1)
    outliers=compute_outliers(data=data, metric=metric, factor=factor)
    for point in outliers:
        plt.plot(point[0], point[1], "o", color="red", markersize=1.5)
    plt.title(f"outliers in red, {metric} metric")
    plt.savefig(f"outliers, metric={metric}.pdf")
    plt.close()

def main():
    data = np.load("data.npy")

    # plot raw data
    plt.savefig("scatter plot.pdf")

    # plot outliers with different parameters
    plot_data_and_outliers(data=data, metric="euclidean", factor=3)
    plot_data_and_outliers(data=data, metric="custom", factor=2.5)

if __name__ == "__main__":
    main()
