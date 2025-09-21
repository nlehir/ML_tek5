"""
    Generate the data used in the kmeans exercise
"""

import os

import matplotlib.pyplot as plt
import numpy as np

mean_1 = (600, 2, 10)
std_1 = 50

mean_2 = (200, 800, 45)
std_2 = 50

mean_3 = (-400, -1000, -100)
std_3 = 50

mean_4 = (-800, 300, -600)
std_4 = 50

mean_5 = (1000, 100, 800)
std_5 = 50

# mean_6 = (1100, 150, 800)
# std_6 = 50

n_cluster = 200

# generate the data
data_1 = np.random.normal(loc=mean_1, scale=std_1, size=(n_cluster, 3))
data_2 = np.random.normal(loc=mean_2, scale=std_2, size=(n_cluster, 3))
data_3 = np.random.normal(loc=mean_3, scale=std_3, size=(n_cluster, 3))
data_4 = np.random.normal(loc=mean_4, scale=std_4, size=(n_cluster, 3))
data_5 = np.random.normal(loc=mean_5, scale=std_5, size=(n_cluster, 3))
# data_6 = np.random.normal(loc=mean_6, scale=std_6, size=(n_cluster, 3))
data = np.concatenate((data_1, data_2))
data = np.concatenate((data, data_3))
data = np.concatenate((data, data_4))
data = np.concatenate((data, data_5))
# data = np.concatenate((data, data_6))

fig = plt.figure()
ax = plt.axes(projection="3d")

# ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');
x_points = data[:, 0]
y_points = data[:, 1]
z_points = data[:, 2]

ax.scatter3D(x_points, y_points, z_points, alpha=0.3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# plot the data
plt.savefig(os.path.join("images", "data_3D.pdf"))
plt.close()

# save the data
datapath = os.path.join("data", "data_3D.npy")
np.save(datapath, data)
