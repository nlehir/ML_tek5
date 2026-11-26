import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


def main() -> None:
    digits = load_digits()

    # the data consists in 1797 samples
    # of 8*8 pixels images

    pca = PCA().fit(digits.data)

    variance_ratio = pca.explained_variance_ratio_
    """
    Add lines here
    """


if __name__ == "__main__":
    main()
