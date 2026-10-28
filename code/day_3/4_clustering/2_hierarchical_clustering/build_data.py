import numpy as np

nb_persons_1 = 20
nb_persons_2 = 15
nb_persons_3 = 15

mean_1 = (3, 4)
mean_2 = (10, 2)
mean_3 = (5, 15)

std_1 = 0.5
std_2 = 1
std_3 = 2

# std_1 = 0.05
# std_2 = 0.1
# std_3 = 0.2

group_1 = np.random.normal(mean_1, std_1, (nb_persons_1, 2))
group_2 = np.random.normal(mean_2, std_2, (nb_persons_2, 2))
group_3 = np.random.normal(mean_3, std_3, (nb_persons_3, 2))

data = np.concatenate((group_1, group_2))
data = np.concatenate((data, group_3))

np.savetxt("addresses.csv", data, delimiter=",")
