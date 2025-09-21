import math
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

threshold = 1
overlap = 0.1

N_SAMPLES = 400


def inside(threshold):
    """
    Generate data closer to the origin
    """
    radius = np.random.uniform(0, (1 + overlap) * threshold, N_SAMPLES // 2)
    theta = np.random.uniform(0, 2 * math.pi, N_SAMPLES // 2)
    Xpos = radius * np.cos(theta)
    ypos = radius * np.sin(theta)
    return Xpos, ypos


def outside(threshold):
    """
    Generate data further from the origin
    """
    radius = np.random.uniform((1 - overlap) * threshold, 3 * threshold, N_SAMPLES)
    theta = np.random.uniform(0, 2 * math.pi, N_SAMPLES)
    Xpos = radius * np.cos(theta)
    ypos = radius * np.sin(theta)
    return Xpos, ypos


def main():
    Xposin, yposin = inside(1)
    Xposout, yposout = outside(1.1)

    X_in = np.column_stack((Xposin, yposin))
    y_in = np.zeros((len(X_in), 1))
    X_out = np.column_stack((Xposout, yposout))
    y_out = np.ones((len(X_out), 1))
    X = np.vstack((X_in, X_out))
    y = np.vstack((y_in, y_out))

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train_0 = X_train[np.where(y_train == 0)[0]]
    X_train_1 = X_train[np.where(y_train == 1)[0]]
    X_test_0 = X_test[np.where(y_test == 0)[0]]
    X_test_1 = X_test[np.where(y_test == 1)[0]]

    plt.plot(
        X_train_0[:, 0],
        X_train_0[:, 1],
        "o",
        label="class 0 train",
        alpha=0.5,
        color="skyblue",
    )
    plt.plot(
        X_test_0[:, 0],
        X_test_0[:, 1],
        "o",
        label="class 0 test",
        alpha=1,
        color="skyblue",
    )
    plt.plot(
        X_train_1[:, 0],
        X_train_1[:, 1],
        "o",
        label="class 1 train",
        alpha=0.5,
        color="mediumblue",
    )
    plt.plot(
        X_test_1[:, 0],
        X_test_1[:, 1],
        "o",
        label="class 1 test",
        alpha=1,
        color="mediumblue",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xticks([-3, 0, 3])
    plt.yticks([-3, 0, 3])
    plt.legend(loc="best")
    plt.title("classification problem")
    plt.savefig(os.path.join("images", "data.pdf"))

    np.save(os.path.join("data", "X_train"), X_train)
    np.save(os.path.join("data", "X_test"), X_test)
    np.save(os.path.join("data", "y_train"), y_train)
    np.save(os.path.join("data", "y_test"), y_test)


if __name__ == "__main__":
    main()
