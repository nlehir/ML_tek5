import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from utils import prediction_squared_error

# load data
X_path = os.path.join("data", "X.npy")
y_path = os.path.join("data", "y.npy")
X = np.load(X_path)
y = np.load(y_path)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


def objective(trial):
    """
    Objective function:
        Generate a set of hyperparameters, fit a ridge regressor
        on the train set using this set of parameters, and return the 
        r2 score on the test set.

        Note that using only a train / test dataset splitting strategy is not
        always the best approach, and one might need to use a validation set,
        or even cross-validation. However, cross-validation requires more
        computation time (see the class). hence the most relevant approach
        always depends on the context.

    The quantity returned by this function (here, the r2 score on the test set) will depend on
    the hyperparameters, that will be generated inside the function itsself.
    Optuna handles the hyperparameter generation process and uses the final
    score in order to choose which hyperparameter values to try next.

    Fix this function by using the optuna API.
    https://optuna.org/
    """
    return 1

def main():
    # database for the optuna dashboard
    storage_name = "ridge.db"
    # clean study if exists
    # you may not want this behavior in practical applications
    if os.path.exists(storage_name):
        os.remove(storage_name)
    # create a study
    study = optuna.create_study(
        storage=f"sqlite:///{storage_name}",
        study_name="Ridge_regression",
        load_if_exists=False,
        direction="maximize",  # we want to maximize the R2 score
    )
    study.optimize(objective, n_trials=50)

    # print best trial
    print(f"Best value: {study.best_value} (params: {study.best_params})")
    for key, value in study.best_trial.params.items():
        if type(value) == float:
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # extract dataframe
    df = study.trials_dataframe()

    """
    Use the dataframe to plot the influence of the parameters on the R2 score
    """


if __name__ == "__main__":
    main()
