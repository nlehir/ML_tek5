import numpy as np

nb_points = 10000

data_x = np.random.normal(30, 1, (nb_points))
data_y = np.random.normal(60, 10, (nb_points))

data = np.column_stack((data_x, data_y))
np.save("data.npy", data)
