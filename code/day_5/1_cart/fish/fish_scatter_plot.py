import numpy as np
import matplotlib.pyplot as plt

fish_features = np.load("data/fish_features.npy")
fish_class = np.load("data/fish_class.npy")

feature_names = ["length", "weight"]
class_names = ["tuna", "salmon"]

tuna_index = np.where(fish_class == 0)[0]
salmon_index = np.where(fish_class == 1)[0]

tuna_length = fish_features[tuna_index, 0]
salmon_length = fish_features[salmon_index, 0]

tuna_weight = fish_features[tuna_index, 1]
salmon_weight = fish_features[salmon_index, 1]

plt.plot(tuna_length, tuna_weight, 'o', color="green", alpha=0.5, label="tuna")
plt.plot(salmon_length, salmon_weight, 'o', color="blue", alpha=0.5,
         label="salmon")
plt.xlabel("fish length")
plt.ylabel("fish weight")
plt.legend(loc="best")
plt.savefig("images/visualizations/fish_scatter plot.pdf")
