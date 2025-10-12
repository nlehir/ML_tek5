"""
    add another feature to the dataset
"""
# create a simple fish dataset
import numpy as np

nb_data = 400

# length in centimeters
tuna_length = np.random.normal(45, 6, (nb_data,))
salmon_length = np.random.normal(30, 6, (nb_data,))
fish_length = np.concatenate((tuna_length, salmon_length))

# weight in kilograms
tuna_weight = np.random.normal(6, 3, (nb_data,))
salmon_weight = np.random.normal(4, 1, (nb_data,))
fish_weight = np.concatenate((tuna_weight, salmon_weight))

#  birth hour
salmon_birth_hour = np.random.uniform(0, 10, (nb_data,))
tuna_birth_hour = np.random.uniform(0, 10, (nb_data,))
fish_birth_hour = np.concatenate((tuna_birth_hour, salmon_birth_hour))

# put all the features together
fish_features = np.column_stack((fish_length, fish_weight))
fish_features = np.column_stack((fish_features, fish_birth_hour))
fish_class = np.concatenate((np.zeros(nb_data), np.ones(nb_data)))

# save the data to python files
np.save("data/fish_features_blurred", fish_features)
np.save("data/fish_class_blurred", fish_class)
