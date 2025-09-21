"""
Exercise illustrating hierarchical clustering.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from hierarchical_clustering import find_closest_classes

# from hierarchical_clustering_solution import find_closest_classes
from sklearn import metrics

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

metric = "euclidean"

# build distance matrix between points
condensed_distance = scipy.spatial.distance.pdist(data, metric=metric)
distance_matrix = scipy.spatial.distance.squareform(condensed_distance)

# initializes classes
# each data point is its own class.
nb_datapoints = x_coordinates.shape[0]
classes = [[i] for i in range(nb_datapoints)]


def get_label(sample: int, classes: list[list]) -> int:
    for class_id, class_content in enumerate(classes):
        if sample in class_content:
            return class_id


def main():
    silhouette_scores = list()
    nb_clusters = list()
    step = 0
    LINKAGE = "single"
    print("hierarchical clustering")
    print(f"use {LINKAGE} linkage")
    while len(classes) > 2:
        print(f"step: {step}")
        step += 1
        index_1, index_2 = find_closest_classes(classes, linkage=LINKAGE)
        classes[index_1] = classes[index_1] + classes[index_2]
        classes.remove(classes[index_2])
        labels = [get_label(i, classes) for i in range(nb_datapoints)]
        silhouette_score = metrics.silhouette_score(data, labels, metric="euclidean")
        silhouette_scores.append(silhouette_score)
        nb_clusters.append(len(classes))
    plt.plot(nb_clusters[-10:], silhouette_scores[-10:], "o")
    plt.xlabel("number of clusters")
    plt.ylabel("silhouette score")
    plt.title(f"silhouette scores\n{LINKAGE} linkage")
    fig_path = os.path.join("images", f"silhouette_scores_{LINKAGE}_linkage.pdf")
    plt.savefig(fig_path)
    plt.close()


if __name__ == "__main__":
    main()
