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

"""
add lines here
"""
