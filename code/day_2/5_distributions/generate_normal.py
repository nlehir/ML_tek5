"""
Create a dummy csv
"""

import csv

import numpy as np

file_name = "distribution_1.csv"

mean = 3
std_dev = 2
nb_point = 100000

with open("csv_files/" + file_name, "w") as csvfile:
    filewriter = csv.writer(csvfile, delimiter=",")
    for point in range(1, nb_point):
        random_variable = np.random.normal(loc=mean, scale=std_dev)
        filewriter.writerow([str(point), str(random_variable)])
