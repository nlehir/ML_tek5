import numpy as np
from sklearn.metrics import r2_score


def mean_squared_error(
    x: np.ndarray, y: np.ndarray, w1: np.ndarray, w2: np.ndarray
) -> float:
    hh = x @ w1
    h_relu = np.maximum(hh, 0)
    y_pred = h_relu @ w2
    loss = ((y_pred - y)**2).sum() / len(y)
    return loss

def nn_r2(
    x: np.ndarray, y: np.ndarray, w1: np.ndarray, w2: np.ndarray
) -> float:
    hh = x @ w1
    h_relu = np.maximum(hh, 0)
    y_pred = h_relu @ w2
    return r2_score(y_true=y, y_pred=y_pred)
