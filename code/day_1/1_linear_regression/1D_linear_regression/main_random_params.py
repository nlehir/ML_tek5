"""
Evaluate the quality of randomly drawn parameters for a
1 dimensional linear regression.
"""

import os

import numpy as np

from utils import empirical_risk
# from utils_solution import empirical_risk

from constants import STD_NOISE
from utils_files import load_data

def main() -> None:
    n_tests = 100

    X_train, X_test, y_train, y_test = load_data(std=STD_NOISE)

    rng = np.random.default_rng()

    best_train_error = 10e12
    best_theta = 0
    best_b = 0

    for test_id in range(n_tests):
        theta = rng.uniform(-100, 100)
        b = rng.uniform(-100, 100)
        train_error = empirical_risk(
                theta=theta,
                b=b,
                X=X_train,
                y=y_train,
                )
        if train_error < best_train_error:
            best_train_error = train_error
            best_theta = theta
            best_b = b
        print(
            f"test:             {test_id}"
            f"\ntheta:            {theta:.2f}"
            f"\nb:                {b:.2f}"
            f"\nempirical risk:   {train_error:.2E}\n"
        )
    test_error = empirical_risk(
            theta=best_theta,
            b=best_b,
            X=X_test,
            y=y_test,
            )
    print(
        "\n--------"
        f"\nbest theta:          {best_theta:.2f}"
        f"\nbest b:              {best_b:2f}"
        f"\nbest train error: {best_train_error:.2E}\n"
        f"\nbest error: {test_error:.2E}\n"
    )


if __name__ == "__main__":
    main()
