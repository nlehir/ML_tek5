import os

import numpy as np
import torch


def load_data(folder):
    X_train = np.load(os.path.join(folder, "X_train.npy"))
    X_test = np.load(os.path.join(folder, "X_test.npy"))
    y_train = np.load(os.path.join(folder, "y_train.npy"))
    y_test = np.load(os.path.join(folder, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def clean_filename(name):
    name = name.replace(" ", "_")
    name = name.replace(".", "_")
    return name
