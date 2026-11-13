"""
Perform logistic regression on the digits dataset
As there are 10 classes, it is called a multi class classification problem
"""

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    data = load_digits()
    X = data["data"]
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    """
    Choice of the multi class handling method,
    and the regularization method and magnitude.
    """
    clf = LogisticRegression(
        multi_class="multinomial",
        C=0.01,
    )

    clf.fit(
        X=X_train,
        y=y_train,
    )

    """
    In the multi class case
    """
    print(f"\nTheta shape: {clf.coef_.shape}")
    print(f"Intercept_ shape: {clf.intercept_.shape}")

    """
    Theta for class 0
    """
    print(f"\nTheta for class 0")
    print(clf.coef_[0])
    print("intercept for class 0")
    print(clf.intercept_[0])

    """
    Print scores
    """
    print(f"\ntrain accuracy: {clf.score(X_train, y_train):.4f}")
    print(f"test accuracy: {clf.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
