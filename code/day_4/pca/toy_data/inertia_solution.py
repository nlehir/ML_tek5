"""
compute the inertia
of a dataset related to a given axis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from orthogonal_projection import orthogonal_projection


def test_axis(axis: np.ndarray, data: np.ndarray):
    """
    Evaluates the inertia
    of the dataset related to an axis.
    we assume the data are centered.

    The axis is first encoded as a vector M=(u, v),
    such that the axis corresponds to
    the straight the line (OM).

    u must be nonzero.
    """
    x_data = data[:, 0]
    y_data = data[:, 1]
    # normalize the axis
    axis = 1 / np.linalg.norm(axis) * axis

    # plot the data
    plt.plot(x_data, y_data, "o", color="olivedrab", markersize="3")

    # plot the axis we want to project on
    coefficient_directeur = axis[1] / axis[0]
    x_axis = np.linspace(-8, 8, 100)
    plt.plot(
        x_axis,
        coefficient_directeur * x_axis,
        alpha=0.5,
        color="darkblue",
        label="axis",
    )

    nb_datapoints = len(data)

    """
    project each datapoint to the chosen axis
    and compute the inertia due to this point.
    """

    # vectorized method
    projections = (data @ axis.reshape(2, 1)) * axis
    differences = data - projections
    inertia_2 = np.linalg.norm(differences) ** 2 / nb_datapoints

    # iterative method
    inertia = 0
    for datapoint_index in range(nb_datapoints):
        vector = data[datapoint_index]
        projected_vector = orthogonal_projection(vector, axis)
        # compute the inertia due to this sample
        difference = vector - projected_vector
        difference_norm = np.linalg.norm(difference)
        inertia += difference_norm**2

        # plot the projection
        plt.plot(
            [vector[0], projected_vector[0]],
            [vector[1], projected_vector[1]],
            color="mediumturquoise",
            alpha=0.5,
            label="projected vector",
        )

    inertia /= nb_datapoints
    np.testing.assert_almost_equal(inertia, inertia_2)
    print(inertia)
    # np.testing.assert_equal(inertia, inertia_2)

    title = f"axis=({axis[0]:.2f}, {axis[1]:.2f}) \ninertia = {inertia:.2f}"
    print(f"{title}\n")
    plt.title(title)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    fig_name = f"projection axis=({axis[0]:.2f}, {axis[1]:.2f}).pdf"
    fig_path = os.path.join(
        "images",
        fig_name,
    )
    plt.savefig(fig_path)
    plt.close()


def main() -> None:
    # load and center the data
    data = np.load("data.npy")
    x_data = data[:, 0]
    y_data = data[:, 1]
    x_data = x_data - np.mean(x_data)
    y_data = y_data - np.mean(y_data)
    data = np.column_stack((x_data, y_data))

    # plot the centered data
    plt.plot(x_data, y_data, "o", color="olivedrab", markersize="3")
    plt.title("centered data")
    fig_name = "centered_data"
    fig_path = os.path.join(
        "images",
        fig_name,
    )
    plt.savefig(fig_path)
    plt.close()

    # choose axes and compute the inertia
    axes = [
        np.array([3, 1]),  # 13.738884834780901
        np.array([1, -2]),  # 0.9403642151086317
        np.array([0.1, -2]),  # 0.7420155853224581
        np.array([-2.5, 8]),  # 0.3207911064347376
    ]
    for axis in axes:
        test_axis(
            axis=axis,
            data=data,
        )


if __name__ == "__main__":
    main()
