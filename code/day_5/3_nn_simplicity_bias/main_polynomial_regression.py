"""
Find the optimal parameters for a 1D linear regression
and plot the prediction made by this estimator.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from constants import SIGMA
from utils import load_data, clean_filename


def fit_polynom(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    degree: int,
):
    print(f"fit polynom of degree {degree}")
    """
    Fit a polynom of the degree to the data
    """
    polynom = np.polyfit(x=X_train, y=y_train, deg=degree)

    train_predictions = np.polyval(p=polynom, x=X_train)
    test_predictions = np.polyval(p=polynom, x=X_test)
    X_plot = np.linspace(X_train.min(), X_train.max(), num=200)
    y_plot = np.polyval(p=polynom, x=X_plot)

    """
    Compute the train and test error
    """
    n_train = X_train.shape[0]
    train_error = (np.linalg.norm(train_predictions - y_train) ** 2) / n_train
    n_test = X_test.shape[0]
    test_error = (np.linalg.norm(test_predictions - y_test) ** 2) / n_test

    """
    Plot
    """
    plt.plot(X_train, y_train, "o", alpha=0.7, label="train")
    plt.plot(X_test, y_test, "o", alpha=0.7, label="test")
    plt.plot(X_plot, y_plot, alpha=0.7, label="fitted polynom")

    plt.xlabel("temperature (°C)")
    plt.ylabel("power_consumption (MW)")
    plt.legend(loc="best")
    title = (
        f"Polynomial regression, degree {degree}"
        "\n(empirical risk minimization)"
        f"\ntrain error: {train_error:.2E}"
        f"\ntest error: {test_error:.2E}"
        f"\nnoise standard deviation: {SIGMA:.2E}"
    )
    plt.title(title)
    plt.tight_layout()
    file_name = f"polyomial_regression_std_{SIGMA:.2E}_degree_{degree}"
    file_name = clean_filename(name=file_name)
    fig_path = os.path.join("images", "polynomial_regression", f"{file_name}.pdf")
    plt.savefig(fig_path)
    plt.close()
    return train_error, test_error


def main():
    print(f"Polynomial regression, standard deviation {SIGMA}")
    DATA_FOLDER = os.path.join("data")
    X_train, X_test, y_train, y_test = load_data(folder=DATA_FOLDER)

    degrees = range(2, 30)
    train_errors = list()
    test_errors = list()
    for degree in degrees:
        train_error, test_error = fit_polynom(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            degree=degree,
        )
        train_errors.append(train_error)
        test_errors.append(test_error)

    # plot dataset
    plt.plot(train_errors, "o", alpha=0.7, label="train error")
    plt.plot(test_errors, "o", alpha=0.7, label="test error")
    plt.xlabel("degree")
    plt.ylabel("squared error")
    plt.yscale("log")
    plt.legend(loc="best")
    title = f"Polynomial regression\nnoise standard deviation {SIGMA}"
    plt.title(title)
    plt.tight_layout()
    fig_name = f"polynomial_regression_standard_deviation_{SIGMA}"
    fig_name = f"{clean_filename(fig_name)}.pdf"
    fig_path = os.path.join("images", "polynomial_regression", fig_name)
    plt.savefig(fig_path)


if __name__ == "__main__":
    main()
