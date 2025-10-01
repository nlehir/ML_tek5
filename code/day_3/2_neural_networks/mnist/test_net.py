"""
Use a previously trained model
"""


import matplotlib.pyplot as plt
import os
import numpy as np
from keras.models import load_model
from keras.activations import *


def predict_test_point(
        data_index: int,
        x_test: np.ndarray,
        y_test: np.ndarray,
        loaded_model,
        ) -> None:
    """
    test if the model predicts
    the correct class
    for a given datapoint from
    the mnist database.
    """
    nb_test_data = x_test.shape[0]
    if data_index > nb_test_data:
        raise ValueError(
            f"data index too large : only {nb_test_data} test samples available"
        )

    print(f"test data point {data_index}")
    true_label = y_test[data_index]

    # prediction of the model
    pred = loaded_model.predict(x_test[data_index].reshape(1, 28, 28, 1))
    predicted_label = pred.argmax()
    if true_label == predicted_label:
        print(f"correct prediction: {predicted_label}")
        title = f"testing point: {data_index}\ntrue label: {true_label}\npredicted label: {predicted_label} \nOK"
    else:
        print(f"-- ! wrong prediction: {predicted_label} instead of {true_label}")
        title = f"testing point: {data_index}\ntrue label: {true_label}\npredicted label: {predicted_label}\nMISTAKE"

        image = x_test[data_index][:, :, 0]
        plt.imshow(image, cmap="Greys")
        plt.title(title)
        plt.savefig(os.path.join("images", f"prediction_{data_index}.pdf"))
        plt.close()

def main() -> None:
    y_test = np.load("data/y_test.npy")
    x_test = np.load("data/x_test.npy")
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    loaded_model = load_model("trained_model.keras")

    loaded_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    samples_to_test = np.arange(1, 1000)
    for sample_id in samples_to_test:
        predict_test_point(
                data_index=sample_id,
                x_test=x_test,
                y_test=y_test,
                loaded_model=loaded_model,
                )

if __name__ == "__main__":
    main()
