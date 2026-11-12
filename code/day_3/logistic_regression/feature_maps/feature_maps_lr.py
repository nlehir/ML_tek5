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
    Edit from here: transform the data and apply logistic regression
    to the transformed data
    """


if __name__ == "__main__":
    main()
