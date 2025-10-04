from sklearn.datasets import load_digits
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def main() -> None:
    digits = load_digits()

    # the data consists in 1797 samples
    # of 8*8 pixels images
    print(digits.data.shape)

    # we use PCA in order to project them
    # on principal components
    pca = PCA(n_components=2)
    projected_data = pca.fit_transform(digits.data)

    print(projected_data.shape)

    plt.scatter(
        projected_data[:, 0],
        projected_data[:, 1],
        c=digits.target,
        edgecolor="none",
        alpha=0.5,
        cmap=plt.cm.get_cmap("jet", 10),
    )
    title = (
            "Digits projected on 2 dimensions"
            " after PCA\n"
            "the colors correspond to the labels"
            " but are not used by the PCA"
            )
    plt.title(title, fontsize=9)
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.colorbar()

    fig_name = "projected_digits.pdf"
    fig_path = os.path.join("images", fig_name)
    plt.savefig(fig_path)

if __name__ == "__main__":
    main()
