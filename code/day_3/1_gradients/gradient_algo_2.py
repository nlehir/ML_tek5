"""
Perform gradient descent on a non convex function
"""

import os
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import math


def function_to_minimize(x: float, y: float) -> float:
    """
    simple non convex function that is 
    to be minimized by gradient descent
    """
    return (
        0.5 * y
        + 0.1 * (x**2 + y**2)
        - 100 * math.exp(-((x - 11.5) ** 2 + (y - 8.2) ** 2) / 5)
        - 200 * math.exp(-((x + 9) ** 2 + (y + 11) ** 2) / 10)
    )


def xgradient(x: float, y: float) -> float:
    """
    compute the x coordinate of the gradient
    """
    return (
        0.2 * x
        + 100 * math.exp(-((x - 11.5) ** 2 + (y - 8.2) ** 2) / 5) * (2 / 5) * (x - 11.5)
        + 200 * math.exp(-((x + 9) ** 2 + (y + 11) ** 2) / 10) * (2 / 10) * (x + 9)
    )


def ygradient(x: float, y: float) -> float:
    """
    compute the y coordinate of the gradient
    """
    return (
        0.5
        + 0.2 * y
        + 100 * math.exp(-((x - 11.5) ** 2 + (y - 8.2) ** 2) / 5) * (2 / 5) * (y - 8.2)
        + 200 * math.exp(-((x + 9) ** 2 + (y + 11) ** 2) / 10) * (2 / 10) * (y + 11)
    )

def main() -> None:
    # clean folder that contains the images
    for image in os.listdir("function_2/"):
        os.remove(os.path.join("function_2", image))


    # plot the function to minimize
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    X, Y, Z = axes3d.get_test_data(0.05)
    S = X + Y

    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            S[i][j] = function_to_minimize(X[i][j], Y[i][j])

    ax.plot_wireframe(X, Y, S, rstride=5, cstride=5, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("function_to_minimize_2.pdf")

    # initialize the optimization
    scope = 30
    rng = np.random.default_rng()
    x_iter = rng.uniform(low=-scope, high=scope)
    y_iter = rng.uniform(low=-scope, high=scope)
    N_iterations = 10000

    # set learning rate
    gamma = 0.005

    # initialize lists to store the results
    x_iter_store = list()
    y_iter_store = list()
    z_iter_store = list()

    # iterative optimization
    for iteration in range(N_iterations):
        _xgradient = xgradient(x_iter, y_iter)
        _ygradient = ygradient(x_iter, y_iter)
        x_iter = x_iter - gamma * _xgradient
        y_iter = y_iter - gamma * _ygradient
        z = function_to_minimize(x_iter, y_iter)
        # do not plot all the iterations
        if iteration < 40000:
            if iteration % 1000 == 0:
                x_iter_store.append(x_iter)
                y_iter_store.append(y_iter)
                z_iter_store.append(z)
                ax.plot(x_iter_store, y_iter_store, z_iter_store, color="darkred")
                plt.savefig(f"function_2/{iteration}.pdf")
                print(f"it {iteration}, x : {x_iter:.2f} y : {y_iter:.2f}  value : {z:.2f}")
        elif iteration < 35000:
            if iteration % 100 == 0:
                x_iter_store.append(x_iter)
                y_iter_store.append(y_iter)
                z_iter_store.append(z)
                ax.plot(x_iter_store, y_iter_store, z_iter_store, color="darkred")
                plt.savefig(f"function_2/{iteration}.pdf")
                print(f"it {iteration}, x : {x_iter:.2f} y : {y_iter:.2f}  value : {z:.2f}")
        else:
            if iteration % 50 == 0:
                x_iter_store.append(x_iter)
                y_iter_store.append(y_iter)
                z_iter_store.append(z)
                ax.plot(x_iter_store, y_iter_store, z_iter_store, color="darkred")
                plt.savefig(f"function_2/{iteration}.pdf")
                text = (
                    f"it {iteration}"
                    f", x : {x_iter:.2f} y : {y_iter:.2f}"
                    f"value : {z:.2f}"
                    )

                print(text)

if __name__ == "__main__":
    main()
