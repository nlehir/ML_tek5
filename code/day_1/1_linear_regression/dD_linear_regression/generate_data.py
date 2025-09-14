"""
Generate toy data for d dimensional
linear regression
"""

import os

import numpy as np


def generate_input_data(sigma: float, n: int, d: int, r) -> np.ndarray:
    """
    Generate design matrix
    """
    X = r.rand(n, d - 1)
    X_last_column = X[:, -1].reshape(n, 1)
    noise = r.normal(0, sigma, size=(X_last_column.shape))
    X_added_column = X_last_column + noise
    X = np.hstack((X, X_added_column))
    return X


def generate_output_data(X, theta_star, sigma, r):
    """
    generate output data (supervised learning)
    according to the linear model, fixed design setup
    - X is fixed
    - Y is random, according to

    Y = Xtheta_star + epsilon

    where epsilon is a centered gaussian noise vector with variance
    sigma*In

    Parameters:
        X (float matrix): (n, d) design matrix
        theta_star (float vector): (d, 1) vector (optimal parameter)
        sigma (float): variance each epsilon

    Returns:
        Y (float matrix): output vector (n, 1)
    """

    # output data
    n = X.shape[0]
    noise = r.normal(0, sigma, size=(n, 1))
    y = X @ theta_star + noise
    return y


def main():
    # number of samples and dimension
    n = 400
    d = 200

    #  variance
    sigma = 2

    # use a seed to have consistent resutls
    r = np.random.RandomState(4)

    # generate input data
    X = generate_input_data(sigma, n, d, r)
    X_path = os.path.join("data", "X")
    np.save(X_path, X)

    # generate theta_star
    theta_star = r.rand(d, 1)
    theta_star_path = os.path.join("data", "theta_star")
    np.save(theta_star_path, theta_star)

    # generate output data
    y = generate_output_data(X, theta_star, sigma, r)
    y_path = os.path.join("data", "y")
    np.save(y_path, y)


if __name__ == "__main__":
    main()
