"""
    Generate the data used in the kmeans exercise
"""

import os

import matplotlib.pyplot as plt
import numpy as np

mean_1 = (600, 2)
std_1 = 100

mean_2 = (200, 100)
std_2 = 50

mean_3 = (-400, 100)
std_3 = 100

n_cluster = 2000

# generate the data
data_1 = np.random.normal(loc=mean_1, scale=std_1, size=(n_cluster, 2))
data_2 = np.random.normal(loc=mean_2, scale=std_2, size=(n_cluster, 2))
data_3 = np.random.normal(loc=mean_3, scale=std_3, size=(n_cluster, 2))
data = np.concatenate((data_1, data_2))
data = np.concatenate((data, data_3))

# plot the data
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, "o")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(os.path.join("images", "data_2D.pdf"))
plt.close()

# save the data
datapath = os.path.join("data", "data_2D.npy")
np.save(datapath, data)
