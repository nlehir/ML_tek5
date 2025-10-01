import os

import numpy as np
import matplotlib.pyplot as plt

X_START = 10
GAMMA = 0.01
N_ITER = int(2e2)

def func(x):
    return (x-1)**2 +3.5

def func_derivative(x):
    return 2 * (x-1)


def main() -> None:

    min_x = X_START
    max_x = X_START

    # perform gradient descent
    bbox = dict(boxstyle="circle", fc="b", alpha=0.2)
    x = X_START
    for it in range(N_ITER):
        print(f"\niteration: {it}")
        y = func(x)
        # plot the current iterate
        plt.annotate(text=f"{it}", xy=(x,y), bbox=bbox)
        # gradient update
        x = x - GAMMA * func_derivative(x)
        print(f"x: {x}")
        print(f"f(x): {func(x)}")

        # update the plot range
        if x > max_x:
            max_x = x
        if x < min_x:
            min_x = x


    # plot a little wider
    min_x -= 2
    max_x += 2
    # plot the function on a regular grid
    xx = np.linspace(min_x, max_x)
    yy = func(xx)
    plt.plot(xx, yy, label="function")

    # finish plot
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")
    title=(
            f"gradient descent"
            f"\ngamma={GAMMA}"
            f"\nx_0={X_START}"
            )
    plt.title(title)
    figpath = os.path.join("1d_function", f"gd_gamma_{GAMMA}_x0_{X_START:.1E}.pdf")
    plt.savefig(figpath)
    plt.close()


if __name__ == "__main__":
    main()
