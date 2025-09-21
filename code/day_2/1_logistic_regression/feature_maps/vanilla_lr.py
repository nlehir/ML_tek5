import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from utils_files import load_data


def main():
    # split the X into training and test
    X_train, X_test, y_train, y_test = load_data()
    clf = LogisticRegression().fit(X_train, y_train.ravel())

    print(f"train accuracy: {clf.score(X_train, y_train):.4f}")
    print(f"test accuracy: {clf.score(X_test, y_test):.4f}")

    # print("\ntheta")
    # print(clf.coef_[0])
    # print("intercept")
    # print(clf.intercept_)

    # plot data
    X_train_0 = X_train[np.where(y_train == 0)[0]]
    X_train_1 = X_train[np.where(y_train == 1)[0]]
    X_test_0 = X_test[np.where(y_test == 0)[0]]
    X_test_1 = X_test[np.where(y_test == 1)[0]]
    plt.plot(X_train_0[:, 0], X_train_0[:, 1], "o", label="class 0 train", alpha=0.5, color="skyblue")
    plt.plot(X_test_0[:, 0], X_test_0[:, 1], "o", label="class 0 test", alpha=1, color="skyblue")
    plt.plot(X_train_1[:, 0], X_train_1[:, 1], "o", label="class 1 train", alpha=0.5, color="mediumblue")
    plt.plot(X_test_1[:, 0], X_test_1[:, 1], "o", label="class 1 test", alpha=1, color="mediumblue")
    plt.title("Train set and test set")
    plt.legend(loc="best")

    """
    plot the obtained separator on the same graph
    get the parameters of the separator
    """
    a_1, a_2 = clf.coef_[0]
    b = clf.intercept_
    min_x_data = min(X_train[:, 0])
    max_x_data = max(X_train[:, 0])
    # generate data on the x axis
    xx = np.linspace(min_x_data, max_x_data)
    # compute the y values of the separator
    yy = (-b-a_1*xx)/a_2
    # plot the separator
    plt.plot(xx, yy, label="separator")
    title = (
            "Separation obtained by raw logistic regression"
            f"\ntrain accuracy: {clf.score(X_train, y_train):.4f}"
            f"\ntest accuracy: {clf.score(X_test, y_test):.4f}"
            )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join("images", "vanilla_lr.pdf"))

    print("\nPredictions on the test set")
    print(clf.predict(X_test))


if __name__ == "__main__":
    main()
