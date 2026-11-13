import csv
import os

import ipdb
import numpy as np

# We will create artificial data to make a neural network try to learn them
# However the data wont be random

input_dimension = 6
output_dimension = 2
batch_size = 64
test_size = 32
# we will add some noise to the data
noise_std_dev = 1
noise_mean = 0


def input_output_function(x):
    """
    Our neural network will try to approximate this function
    based on training samples
    """
    noise = np.random.normal(loc=noise_mean, scale=noise_std_dev, size=2)
    return x[3] - x[1] + 2 * x[4] + noise[0], 7 * x[0] - 4 * x[2] + x[5] + noise[1]


inputs_training = np.random.normal(0, 5, (batch_size, input_dimension))
inputs_test = np.random.normal(0, 5, (test_size, input_dimension))

outputs_training = [
    input_output_function(inputs_training[datapoint]) for datapoint in range(batch_size)
]
# convert the outputs to a numpy array
outputs_training = np.asarray(outputs_training)

outputs_test = [
    input_output_function(inputs_test[datapoint]) for datapoint in range(test_size)
]
# convert the outputs to a numpy array
outputs_test = np.asarray(outputs_test)


# save the data to numpy arrays
if not os.path.exists("./data/"):
    os.makedirs("./data/")
np.save(f"data/training_inputs_std_{noise_std_dev}.npy", inputs_training)
np.save(f"data/test_inputs_std_{noise_std_dev}.npy", inputs_test)
np.save(f"data/training_outputs_std_{noise_std_dev}.npy", outputs_training)
np.save(f"data/test_outputs_std_{noise_std_dev}.npy", outputs_test)
