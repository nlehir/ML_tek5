"""
    Exercise illustrating hierarchical clustering.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from sklearn.cluster import linkage_tree

# open file
file_name = "addresses.csv"

x_coordinates = list()
y_coordinates = list()

# load the data
with open(file_name, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        x_coordinates.append(float(row[0]))
        y_coordinates.append(float(row[1]))

x_coordinates = np.asarray(x_coordinates)
y_coordinates = np.asarray(y_coordinates)
data = np.column_stack((x_coordinates, y_coordinates))
nb_datapoints = len(data)

"""
    Show a scatter plot of the data
"""

# third method : matplotlib function
plt.scatter(x_coordinates, y_coordinates)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
fig_path = os.path.join("images", "scatter_plot_matplotlib.pdf")
plt.savefig(fig_path)
plt.close()


"""
    Hierarchical clustering.
    We will use agglomerative clustering.
"""
# choose metric
metric = "euclidean"

# build distance matrix between points
condensed_distance = scipy.spatial.distance.pdist(data, metric=metric)
distance_matrix = scipy.spatial.distance.squareform(condensed_distance)

# initializes classes
# each data point is its own class.
classes = [[i] for i in range(nb_datapoints)]


def find_closest_classes(classes: list[list], linkage: str = "single") -> tuple:
    """
    In the list of classes, find the two closest classes

    :param classes (list): list of clsses
    :param linkage: choice of the linkage method (single, average, etc)

    :returns returned_classes (tuple): two closest classes
    """
    min_dist = np.max(distance_matrix)
    returned_classes = (0, 1)
    """
    add lines here
    """
    return returned_classes


def plot_clustering(step: int, classes: list[list], linkage: str) -> None:
    """
    Plots the clustering resutls
    """
    plt.plot(x_coordinates, y_coordinates, "o")
    for index in range(len(classes)):
        class_to_plot = classes[index]
        x_coord = x_coordinates[class_to_plot]
        y_coord = y_coordinates[class_to_plot]
        plt.plot(x_coord, y_coord, "o")
    plt.title(
        "hierarchical clustering"
        f"\nstep {step}"
        f"\n{len(classes)} clusters"
        f"\n{metric} metric"
        f"\n{linkage} linkage"
    )
    plt.tight_layout()
    fig_name = f"{linkage}_linkage_step_{step}.pdf"
    fig_path = os.path.join("clusterings", fig_name)
    plt.savefig(fig_path)
    plt.close()


def main():
    step = 0
    LINKAGE = "single"
    print("hierarchical clustering")
    print(f"use {LINKAGE} linkage")
    while len(classes) > 1:
        print(f"step: {step}")
        step += 1
        index_1, index_2 = find_closest_classes(classes, linkage=LINKAGE)
        classes[index_1] = classes[index_1] + classes[index_2]
        classes.remove(classes[index_2])
        plot_clustering(step, classes, linkage=LINKAGE)


if __name__ == "__main__":
    main()
