import os

import matplotlib.pyplot as plt
import torch
from constants import SIGMA
from utils import clean_filename, load_data

# load data
DATA_FOLDER = os.path.join("data")
X_train, X_test, y_train, y_test = load_data(folder=DATA_FOLDER)

LEARNING_RATES = [10 ** (-k) for k in range(3, 7)]
DIM_H = [10, 20, 30, 40, 100, 200]

X_train = X_train.reshape(len(X_train), 1)
y_train = y_train.reshape(len(y_train), 1)
X_test = X_test.reshape(len(X_test), 1)
y_test = y_test.reshape(len(y_test), 1)
X_train = torch.tensor(X_train).type(torch.float)
X_test = torch.tensor(X_test).type(torch.float)
y_train = torch.tensor(y_train).type(torch.float)
y_test = torch.tensor(y_test).type(torch.float)
n_train_samples, dim_in = X_train.shape
dim_out = y_train.shape[1]

# overparametrized
print(f"n_train_samples: {n_train_samples}")
print(f"dim_in: {dim_in}")
print(f"dim_out: {dim_out}")

# loss_fn = torch.nn.MSELoss(reduction="sum")
loss_fn = torch.nn.MSELoss()
