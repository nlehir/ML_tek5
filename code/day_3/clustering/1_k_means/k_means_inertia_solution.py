"""
Assess the quality of the clustering using the inertia knee criterion
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans

# load the data
datapath = os.path.join("data", "data_3D.npy")
data = np.load(datapath)

nbs_of_clusters = range(1, 15)
inertias = list()

for nb_of_clusters in nbs_of_clusters:
    kmeans = KMeans(n_clusters=nb_of_clusters).fit(data)
    inertia = kmeans.inertia_
    print(f"{nb_of_clusters} clusters: inertia = {inertia:.2E}")
    inertias.append(inertia)

# plot the data and the found centroids
plt.plot(nbs_of_clusters, inertias)
plt.xlabel("number of centroids")
plt.xticks(range(1, 13))
plt.ylabel("inertia")
plt.savefig("inertia.pdf")
plt.close()

"""
    Use an algorithm to find the knee (maximum curvature)
    https://github.com/arvkevi/kneed
"""
kneedle = KneeLocator(
    nbs_of_clusters, inertias, S=1.0, curve="convex", direction="decreasing"
)
print(f"\nknee at {kneedle.knee} clusters\n")
print(kneedle.knee_y)
