import numpy as np


def empirical_risk(
    theta: float,
    b: float,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Compute the empirical risk of a set of parameters
    for a linear prediction, in 1 dimension.
    """
    n_train = X.shape[0]
    predictions = X * theta + b
    errors = predictions - y
    empirical_risk = (np.linalg.norm(errors) ** 2) / n_train
    """
    Optionally, it for could instance be written as
    (errors ** 2).sum()/n_train
    """
    return empirical_risk
