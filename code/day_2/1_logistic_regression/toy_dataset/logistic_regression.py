"""
perform logistic regression on some toy data, using scikit-learn

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    """
    Load data
    """
    data_path = os.path.join("data", "data.npy")
    labels_path = os.path.join("data", "labels.npy")
    data = np.load(data_path)
    labels = np.load(labels_path)

    """
    Reshape to avoid scikit complaining
    """
    labels = labels.ravel()

    """
    Split the data into training and test
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33)

    """
    Get the dimensions
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    print(f"n train: {n_train}")
    print(f"n test: {n_test}")

    """
    Plot the data
    """
    X_train_1 = X_train[np.where(y_train == 1)[0]]
    X_train_2 = X_train[np.where(y_train == 0)[0]]
    X_test_1 = X_test[np.where(y_test == 1)[0]]
    X_test_2 = X_test[np.where(y_test == 0)[0]]
    plt.plot(
        X_train_1[:, 0],
        X_train_1[:, 1],
        "o",
        label="class 1 train",
        alpha=0.5,
        color="blue",
    )
    plt.plot(
        X_test_1[:, 0], X_test_1[:, 1], "o", label="class 1 test", alpha=1, color="blue"
    )
    plt.plot(
        X_train_2[:, 0],
        X_train_2[:, 1],
        "o",
        label="class 2 train",
        alpha=0.5,
        color="orange",
    )
    plt.plot(
        X_test_2[:, 0],
        X_test_2[:, 1],
        "o",
        label="class 2 test",
        alpha=1,
        color="orange",
    )
    plt.title("Train set and test set")
    plt.legend(loc="best")
    plt.savefig("train_test.pdf")

    """
    Instantiate and optimize a 
    logistic regression estimator
    """
    clf = LogisticRegression().fit(X=X_train, y=y_train)

    """
    Plot the scores
    In scikit, the default classification score is the accuracy
    """
    print("train accuracy")
    print(clf.score(X_train, y_train))
    print("test accuracy")
    print(clf.score(X_test, y_test))
    print("theta")
    print(clf.coef_[0])
    print("intercept")
    print(clf.intercept_)

    # plot the obtained separator on the same graph
    # get the parameters of the separator
    a_1, a_2 = clf.coef_[0]
    b = clf.intercept_
    min_x_data = min(X_train[:, 0])
    max_x_data = max(X_train[:, 0])
    # generate data on the x axis
    xx = np.linspace(min_x_data, max_x_data)
    # compute the y values of the separator
    yy = [(-b - a_1 * x) / a_2 for x in xx]
    # plot the separator
    plt.plot(xx, yy, label="separator")
    plt.title("Separation obtained by logistic regression")
    plt.savefig("separation.pdf")


if __name__ == "__main__":
    main()
