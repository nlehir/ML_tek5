# create a simple fish dataset
import numpy as np
import matplotlib.pyplot as plt

nb_data = 400

# length in centimeters
tuna_length = np.random.normal(45, 6, (nb_data,))
salmon_length = np.random.normal(30, 6, (nb_data,))
fish_length = np.concatenate((tuna_length, salmon_length))

# weight in kilograms
tuna_weight = np.random.normal(7, 1.5, (nb_data,))
salmon_weight = np.random.normal(4, 1, (nb_data,))
fish_weight = np.concatenate((tuna_weight, salmon_weight))

# put all the features together
fish_features = np.column_stack((fish_length, fish_weight))
fish_class = np.concatenate((np.zeros(nb_data), np.ones(nb_data)))

# save the data to python files
np.save("data/fish_features", fish_features)
np.save("data/fish_class", fish_class)

"""
plot histograms
"""
nbins = 50
plt.hist(fish_length, bins=nbins)
title = "distribution of the length of the fish in centimeters"
plt.title(title)
plt.xlabel('value')
plt.ylabel('nb of occurrences')
plt.savefig("images/visualizations/fish_length.pdf")
plt.close()


plt.hist(fish_weight, bins=nbins)
title = "distribution of the weight of the fish in kilos"
plt.title(title)
plt.xlabel('value')
plt.ylabel('nb of occurrences')
plt.savefig("images/visualizations/fish_weight.pdf")
plt.close()

"""
plot histograms with class
"""
plt.hist(tuna_length, color="green", bins=nbins, label="tuna", alpha=0.5)
plt.hist(salmon_length, color="blue", bins=nbins, label="salmon", alpha=0.5)
title = "distribution of the length of the fish in centimeters"
plt.title(title)
plt.xlabel('value')
plt.ylabel('nb of occurrences')
plt.legend(loc="best")
plt.savefig("images/visualizations/fish_length_with_class.pdf")
plt.close()

plt.hist(tuna_weight, color="green", bins=nbins, label="tuna", alpha=0.5)
plt.hist(salmon_weight, color="blue", bins=nbins, label="salmon", alpha=0.5)
title = "distribution of the weight of the fish in Kg"
plt.title(title)
plt.xlabel('value')
plt.ylabel('nb of occurrences')
plt.legend(loc="best")
plt.savefig("images/visualizations/fish_weight_with_class.pdf")
plt.close()
