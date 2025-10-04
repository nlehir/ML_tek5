"""
    adapted from
    https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

    # TODO: improve <01-12-22, nlehir> #
"""
import os
from pandas import core
from sklearn.datasets import load_digits
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


def reconstruct_sample(
    x: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    coefficients: np.ndarray,
    n_components: int,
):
    FONTSIZE = 12
    FONTSIZE_SMALL = 3 * FONTSIZE // 4

    fig = plt.figure(figsize=(1.2 * (5 + n_components), 1.2 * 2))
    g = plt.GridSpec(nrows=2, ncols=5 + n_components, hspace=0.3)

    def show(i, j, x, title=None, fontsize=FONTSIZE):
        ax = fig.add_subplot(g[i, j], xticks=[], yticks=[])
        ax.imshow(x.reshape((8, 8)), interpolation="nearest")
        if title:
            ax.set_title(title, fontsize=fontsize)

    show(i=slice(2), j=slice(2), x=x, title="Sample")

    """
    Start the reconstruction of the original image
    """
    reconstruction = mean.copy()

    show(i=0, j=2, x=reconstruction, title="mean", fontsize=FONTSIZE_SMALL)
    show(i=1, j=2, x=reconstruction, title="1 . mean", fontsize=FONTSIZE_SMALL)

    """
    Iteratively add components to the reconstruction
    """
    for i in range(n_components):
        reconstruction = reconstruction + coefficients[i] * components[i]
        show(
            i=0,
            j=i + 3,
            x=components[i],
            title=f"c_{i}",
            fontsize=FONTSIZE_SMALL,
        )
        show(
            i=1,
            j=i + 3,
            x=reconstruction,
            title=f"{coefficients[i]:.2f} . c_{i+1}",
            fontsize=FONTSIZE_SMALL,
        )
        plt.gca().text(
            0,
            1.05,
            "$+$",
            ha="right",
            va="bottom",
            transform=plt.gca().transAxes,
            fontsize=FONTSIZE,
        )

    show(i=slice(2), j=slice(-2, None), x=reconstruction, title="Reconstruction")
    return fig


def process_sample(
    data_index: int,
    nb_components: int,
    pca,
    digits,
    projected_dataset: np.ndarray,
) -> None:
    print(f"process sample {data_index}")
    sample = digits.data[data_index]
    projected_sample = projected_dataset[data_index]

    fig = reconstruct_sample(
        x=sample,
        coefficients=projected_sample,
        mean=pca.mean_,
        components=pca.components_,
        n_components=nb_components,
    )
    figpath = os.path.join("images", "reconstruction", f"sample_{data_index}_{nb_components}_components.pdf")
    plt.savefig(figpath)


def main() -> None:
    """
    perform PCA
    project and plot the data
    on the principal components
    """
    digits = load_digits()

    NB_COMPONENTS_LIST = [5, 10, 15, 20]
    DATA_INDEX_LIST = [30, 40, 2, 25]
    for nb_components in NB_COMPONENTS_LIST:
        pca = PCA(n_components=nb_components)
        projected_dataset = pca.fit_transform(digits.data)
        print(f"{nb_components} components")
        for data_index in DATA_INDEX_LIST:
            process_sample(
                data_index=data_index,
                nb_components=nb_components,
                pca=pca,
                digits=digits,
                projected_dataset=projected_dataset,
            )


if __name__ == "__main__":
    main()
