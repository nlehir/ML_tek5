import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns


def main() -> None:
    n_samples = 1000
    mu = 5
    std = 10
    data = np.random.normal(loc=mu, scale=std, size=(n_samples,))
    data_2 = np.random.normal(loc=mu-30, scale=std, size=(n_samples,))
    data=np.concatenate((data, data_2))
    np.random.shuffle(data)
    nb_samples_list = range(1, 1000, 10)
    for nb_samples in nb_samples_list:
        print(f"kde with {nb_samples} samples")
        subset = data[:nb_samples+1]
        sns.kdeplot(data=subset)
        title=(
                "kernel density estimation\n"
                f"{nb_samples} samples"
                )
        plt.title(title)
        fig_path = os.path.join("images", f"kde_{nb_samples}_samples.pdf")
        plt.savefig(fig_path)
        plt.close()

if __name__ == "__main__":
    main()
