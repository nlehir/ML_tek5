import os

import numpy as np


def clean_filename(name: str):
    weird_characters = [".", " ", ","]
    for character in weird_characters:
        name = name.replace(character, "_")
    return name


def load_data():
    X_train = np.load(os.path.join("data", f"X_train.npy"))
    X_test = np.load(os.path.join("data", f"X_test.npy"))
    y_train = np.load(os.path.join("data", f"y_train.npy"))
    y_test = np.load(os.path.join("data", f"y_test.npy"))
    return X_train, X_test, y_train, y_test
