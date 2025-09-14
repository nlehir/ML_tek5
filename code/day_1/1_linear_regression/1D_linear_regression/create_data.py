import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from constants import MEAN_NOISE, STD_NOISE, N_SAMPLES

from utils_files import clean_filename


def main():

    print(f"Create dataset witd {STD_NOISE} standard deviation, {N_SAMPLES} samples")

    rng = np.random.default_rng()

    def bayes_predictor(x):
        return -3*x - (x/2)**2 + 500

    # temperature in degree
    temperature = np.random.uniform(-5, 35, N_SAMPLES)

    # power consumption in MW
    power_consumption = bayes_predictor(temperature) + rng.normal(
            loc=MEAN_NOISE,
            scale=STD_NOISE,
            size=N_SAMPLES,
            )

    # plot raw dataset
    plt.plot(temperature, power_consumption, "o", alpha=0.7)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Power Consumption (MW)")
    title = (
            "Temperature (°C) vs power consumption (MW)"
            f"\nnoise standard deviation {STD_NOISE:.1f}"
            )
    plt.title(title)
    file_name = f"dataset_standard_deviation_{STD_NOISE:.1f}"
    plt.savefig(f"{clean_filename(file_name)}.pdf")
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
            temperature,
            power_consumption,
            test_size=0.33,
            )

    # plot dataset with the split
    plt.plot(X_train, y_train, "o", alpha=0.7, label="train set")
    plt.plot(X_test, y_test, "o", alpha=0.7, label="test set")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Power Consumption (MW)")
    title = (
            "Temperature (°C) vs power consumption (MW)"
            f"\nnoise standard deviation {STD_NOISE:.1f}"
            )
    plt.title(title)
    plt.legend(loc="best")
    file_name = f"dataset_standard_deviation_{STD_NOISE:.1f}_splitted"
    plt.savefig(f"{clean_filename(file_name)}.pdf")
    plt.close()

    # save dataset
    np.save(os.path.join("data", clean_filename(f"X_train_{STD_NOISE:.1f}")), X_train)
    np.save(os.path.join("data", clean_filename(f"X_test_{STD_NOISE:.1f}")), X_test)
    np.save(os.path.join("data", clean_filename(f"y_train_{STD_NOISE:.1f}")), y_train)
    np.save(os.path.join("data", clean_filename(f"y_test_{STD_NOISE:.1f}")), y_test)

if __name__ == "__main__":
    main()
