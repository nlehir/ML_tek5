"""
    perform PCA using sklearn
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main() -> None:
    # load and center the data
    data = np.load("data.npy")

    # load the sklearn estimator
    pca = PCA(n_components=2)
    pca.fit(data)

    # principal component obtained by the algorithm
    print("components")
    print(pca.components_)

    # variance carried by those axes
    print(f"\nexplained variance {pca.explained_variance_}")

    # variance ratio carried by those axes
    print(f"\nexplained variance ratio {pca.explained_variance_ratio_}")

if __name__ == "__main__":
    main()
