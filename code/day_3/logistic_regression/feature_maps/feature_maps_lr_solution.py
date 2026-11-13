"""
Perform logistic regression on the tranformed data
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils_files import load_data


def main():
    X_train, X_test, y_train, y_test = load_data()

    """
    Apply feature map
    """
    phi_X_train = (X_train**2).sum(axis=1)
    phi_X_test = (X_test**2).sum(axis=1)

    """
    Avoid scikit complaining
    """
    phi_X_train = phi_X_train.reshape(-1, 1)
    phi_X_test = phi_X_test.reshape(-1, 1)
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    clf = LogisticRegression().fit(phi_X_train, y_train)

    print(f"\ntrain accuracy: {clf.score(phi_X_train, y_train):.4f}")
    print(f"test accuracy: {clf.score(phi_X_test, y_test):.4f}")
    print("\ntheta")
    print(clf.coef_[0])
    print("intercept")
    print(clf.intercept_)
    print("\npredictions on test set")
    print(clf.predict(phi_X_test))


if __name__ == "__main__":
    main()
