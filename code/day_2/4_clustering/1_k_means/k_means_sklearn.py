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
"""
add lines here
"""
