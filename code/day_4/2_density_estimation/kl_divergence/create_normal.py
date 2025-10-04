"""
Empirical distirbution to fit with the KL
"""

import csv
import numpy as np

file_name = 'empirical_distribution.csv'

mean = 40
std_dev = 5
nb_point = 10000

with open(file_name, 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    for point in range(1, nb_point):
        random_variable = np.random.normal(loc=mean, scale=std_dev)
        filewriter.writerow([str(point), str(random_variable)])
