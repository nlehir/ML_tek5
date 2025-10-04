import matplotlib.pyplot as plt
import numpy as np

vector = np.array([1, 2])
axis = np.array([3, 1])
coefficient_directeur = axis[1]/axis[0]
axis = 1/np.linalg.norm(axis)*axis


def orthogonal_projection(vector, axis):
    inner_product = np.dot(vector, axis)
    projected_vector = inner_product*axis
    return projected_vector


# # plot the vector
# plt.plot([0, vector[0]], [0, vector[1]],
#          color="blue",
#          label="vector")
# 
# # plot the axis we want to project on
# x_axis = np.linspace(-4, 4, 100)
# plt.plot(x_axis, coefficient_directeur*x_axis,
#          alpha="0.5",
#          color="teal",
#          label="axis")
# 
# # compute and plot the projection
# projected_vector = orthogonal_projection(vector, axis)
# plt.plot([0, projected_vector[0]],
#          [0, projected_vector[1]],
#          color="orange",
#          label="projected vector")
# 
# plt.xlim([-4, 4])
# plt.ylim([-4, 4])
# plt.legend(loc="best")
# plt.savefig("images/projection.pdf")
# plt.close()
