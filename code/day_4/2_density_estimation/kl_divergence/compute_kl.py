"""
Using the KL divergenve, choose the best model to fit the data
"""

import matplotlib.pyplot as plt
import pandas
from scipy.stats import entropy
import numpy as np



def compute_kl_divergence(mean, std, data, bins):
    """
    We will use gaussian models in this example
    """
    # sample new data from our model
    # we will compare them to the empirical set
    n_samples = len(data)
    rng = np.random.default_rng()
    model_sample = rng.normal(mean, std, n_samples)

    # Compute the histogram of data sampled from your model
    # we must use the same bins as the empirical distribution
    # in order to compare the distributions
    model_hist, _, _ = plt.hist(model_sample, bins=bins, alpha=0.4, label="model")

    # plot the histogram of the empirical data
    # to visually compare the two sample sets (empirical vs model)
    data_hist, _, _ = plt.hist(data, bins=bins, alpha=0.4, label="data")

    # we will compute the KL divergence, but before we will
    # cheat and add artificial samples to each value.
    # if we don't do so, the divergence might be infinite
    # because some probabilities can take a value of 0
    model_hist = model_hist + 1
    data_hist = data_hist + 1

    # compute the kl divergence between the two distributions
    # the entropy function of scipy computes the KL
    # divergence if two distributions are provided
    # (see the scipy doc)
    kl_divergence = entropy(model_hist, data_hist)

    plt.annotate(f"KL={kl_divergence:.3f}",
                 (50, 50),
                 fontsize=(14))
    title = f"model mean : {mean}, model std : {std}"
    plt.title(title)
    print(title)
    print(f"KL divergence: {kl_divergence:.4f}\n")
    plt.legend(loc="best")
    plt.xlabel('age (years)')
    plt.ylabel('nb of occurrences')
    plt.savefig(f"images/model_hist_mean_{mean}_std_{std}.pdf")
    plt.close()


def main() -> None:
    # open file
    file_name = 'empirical_distribution.csv'

    df = pandas.read_csv(file_name)
    data = df["age"].values

    """
    build a histogram of the empirical data
    """
    NBINS = 100
    _, bins, _ = plt.hist(data, bins=NBINS)
    title = f"empirical distribution : histogram, {NBINS} bins"
    plt.title(title)
    plt.xlabel('age (years)')
    plt.ylabel('number of occurrences')
    plt.savefig(f"images/empirical_hist_{NBINS}_bins.pdf")
    plt.close()

    """
    Try models with different parameters
    """
    compute_kl_divergence(mean=35, std=3, data=data, bins=bins)
    compute_kl_divergence(mean=25, std=6, data=data, bins=bins)
    compute_kl_divergence(mean=40, std=5, data=data, bins=bins)
    compute_kl_divergence(mean=40, std=6, data=data, bins=bins)


if __name__ == "__main__":
    main()
