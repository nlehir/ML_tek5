"""
OLS with scikit
"""

import os

import numpy as np
from sklearn.linear_model import LinearRegression


def main():
    X_train = np.load(os.path.join("data", "X_train.npy"))
    y_train = np.load(os.path.join("data", "y_train.npy"))
    X_test = np.load(os.path.join("data", "X_test.npy"))
    y_test = np.load(os.path.join("data", "y_test.npy"))

    X_train = X_train.reshape((-1, 1))
    X_test = X_test.reshape((-1, 1))

    # laod an instance of the LinearRegression class
    regressor = LinearRegression()

    # optimize its parameters according to the data
    regressor.fit(X_train, y_train)

    # print the r2 score
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
    print(f"train r2 score: {regressor.score(X_train, y_train)}")
    print(f"test r2 score: {regressor.score(X_test, y_test)}")

    # print theta
    print(f"{regressor.coef_=}")

    # print b
    print(f"{regressor.intercept_=}")

    # predict on some new inputs
    print(regressor.predict(np.array([1, 2, -5, 25]).reshape(-1, 1)))


if __name__ == "__main__":
    main()
