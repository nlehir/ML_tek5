"""
Perform the k-means algorithm on toy data using scikit-learn
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# load the data
datapath = os.path.join("data", "data_2D.npy")
data = np.load(datapath)

# use sklearn in order to perform the algorithm
kmeans = KMeans(n_clusters=6, random_state=10).fit(data)

# use the learned estimator in order to predict the cluster of news points
print(kmeans.predict([[230, 105], [-500, 200]]))

# plot the data and the found centroids
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, "o")
centroids = kmeans.cluster_centers_
x_centroids = list(centroids[:, 0])
y_centroids = list(centroids[:, 1])
plt.plot(x_centroids, y_centroids, "x", color="orange", label="centroids")
plt.legend(loc="best")
plt.savefig(os.path.join("images", "sklearn_centroids.pdf"))
plt.close()
