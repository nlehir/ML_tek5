from sklearn.datasets import load_digits
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def main() -> None:
    digits = load_digits()
    d = digits.data.shape[1]

    # the data consists in 1797 samples
    # of 8*8 pixels images

    pca = PCA().fit(digits.data)

    threshold = 0.75

    variance_ratio = pca.explained_variance_ratio_

    cumulated_variance = np.cumsum(variance_ratio)

    n_kept_components = np.where(cumulated_variance > threshold)[0][0].item()

    message = (
        f"Cumulated variance"
        f"\n{100 * cumulated_variance[n_kept_components]:.3f}‰ of variance "
        f"kept with "
        f"{n_kept_components}/{d} components"
    )
    print(message)

    plt.plot(cumulated_variance, "o", label="cumulated variance")
    plt.plot(range(d), threshold * np.ones_like(variance_ratio), label=threshold)
    plt.xlabel("number of components")
    plt.ylabel("explained variance")
    plt.title(message)
    plt.legend(loc="best")
    plt.tight_layout()
    fig_name = "explained_variance_ratio.pdf"
    fig_path = os.path.join("images", fig_name)
    plt.savefig(fig_path)


if __name__ == "__main__":
    main()
