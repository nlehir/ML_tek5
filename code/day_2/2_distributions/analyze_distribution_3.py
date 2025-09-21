"""
Read data from a csv
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import os

# open file
file_name = os.path.join("csv_files", "distribution_3.csv")

index = list()
data_1 = list()
data_2 = list()
data_3 = list()
data_4 = list()
data_5 = list()

with open(file_name, "r") as f:
    reader = csv.reader(f)
    row_index = 0
    for row in reader:
        if row_index >= 1:
            index.append(int(row[0]))
            data_1.append(float(row[1]))
            data_2.append(float(row[2]))
            data_3.append(float(row[3]))
            data_4.append(float(row[4]))
            data_5.append(float(row[5]))
        row_index = row_index + 1

data = [data_1, data_2, data_3, data_4, data_5]

cov = np.cov(data)

im = plt.imshow(cov)
plt.colorbar(im)
plt.title("covariance matrix")
plt.savefig("images/covariance.pdf")
plt.close()

l = [(i, j) for i in range(0, 5) for j in range(0, 5)]

correlation_matrix = np.zeros((5, 5))

for (i, j) in l:
    correlation_matrix[i][j] = pearsonr(data[i], data[j])[0]

im = plt.imshow(correlation_matrix, vmin=-1, vmax=1)
plt.colorbar(im)
plt.title("correlation matrix")
plt.savefig("images/correlation_matrix.pdf")
plt.close()
