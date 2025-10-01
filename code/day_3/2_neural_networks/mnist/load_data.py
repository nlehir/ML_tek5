import numpy as np
import os
import tensorflow as tf

def main() -> None:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    np.save(os.path.join("data", "x_train"), x_train)
    np.save(os.path.join("data", "y_train"), y_train)
    np.save(os.path.join("data", "x_test"), x_test)
    np.save(os.path.join("data", "y_test"), y_test)

if __name__ == "__main__":
    main()
