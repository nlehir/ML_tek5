"""
    Principal Component Analysis
    generate redundant data
"""

import numpy as np

nb_points = 100

x_data_1 = np.random.normal(4, 1, nb_points)
x_data_2 = np.random.normal(4, 1, nb_points)
y_data = -3 * x_data_1 + np.random.normal(0, 5, nb_points)
z_data = 2 * y_data + x_data_2 + np.random.normal(0, 4, nb_points)
data_4 = y_data + z_data
data_5 = z_data + 1

data = np.column_stack((x_data_1, x_data_2))
data = np.column_stack((data, y_data))
data = np.column_stack((data, z_data))
data = np.column_stack((data, data_4))
data = np.column_stack((data, data_5))
np.save("redundant_data", data)
