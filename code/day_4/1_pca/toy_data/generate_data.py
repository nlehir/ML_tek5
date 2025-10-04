"""
    generate toy data
    for exercises on
    Principal Component Analysis
"""

import numpy as np
import matplotlib.pyplot as plt


nb_points = 100
x_data = np.random.normal(4, 1, nb_points)
y_data = -3 * x_data + np.random.normal(0, 2, nb_points)
data = np.column_stack((x_data, y_data))
np.save("data", data)

plt.plot(x_data, y_data, "o", color="teal", markersize=3)
plt.xlabel("x data")
plt.ylabel("y data")
plt.title("raw data")
plt.savefig("images/data.pdf")
plt.close()
