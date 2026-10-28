"""
Read data from a csv
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# open file
file_name = "distribution_1.csv"

index = list()
data = list()

with open("csv_files/" + file_name, "r") as f:
    reader = csv.reader(f)
    row_index = 0
    for row in reader:
        if row_index >= 1:
            index.append(int(row[0]))
            data.append(float(row[1]))
        row_index = row_index + 1

"""
plot the raw data
"""
plt.plot(index, data)
title = "data 1"
# plt.title(title)
plt.xlabel("index")
plt.ylabel("value")
plt.savefig("images/distribution_1.pdf")
plt.close()

"""
plot a histogram
"""
# use a relevant number of bins !
nbins = 500
plt.hist(data, bins=nbins)
title = f"distribution 1 : histogram, {nbins} bins"
plt.title(title)
plt.xlabel("value")
plt.ylabel("nb of occurrences")
plt.savefig(f"images/distribution_1_hist_{nbins}_bins.pdf")
plt.savefig(f"images/distribution_1_hist_{nbins}_bins.png")
plt.close()

"""
plot a normalized histogram
"""
plt.hist(data, bins=nbins, density=True)
title = f"distribution 1 : density, {nbins} bins"
plt.title(title)
plt.xlabel("value")
plt.ylabel("density")
plt.savefig(f"images/distribution_1_normed_hist_{nbins}_bins.pdf")

# Fit a normal distribution to the data:
mean, std = norm.fit(data)
print("mean and standard deviation found by fitting a normal distribution")
print(f"mean: {mean}, standard deviation: {std}")
# Finally
# Plot the density function
x = np.linspace(0, 8, 100)
p = norm.pdf(x, mean, std)
plt.plot(x, p, "k", linewidth=2)
title = f"mean = {mean:.2f},  std = {std:.2f}"
plt.title(title)
plt.savefig("images/fitted_distribution_1.pdf")
plt.close()
