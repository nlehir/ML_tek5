"""
OLS with scikit
"""

import os

import numpy as np
from sklearn.linear_model import LinearRegression

from constants import STD_NOISE
from utils_files import load_data


def main():
    X_train, X_test, y_train, y_test = load_data(std=STD_NOISE)

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
