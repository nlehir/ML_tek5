import matplotlib.pyplot as plt
import numpy as np

nb_data = 100000
mean = 1
# standard deviation
std = 0.1

style = {"facecolor": "blue", "alpha": 0.2, "pad": 10}
x = np.random.normal(loc=mean, scale=std, size=nb_data)
plt.plot(range(nb_data), x, "o")
title = f"Normal distribution, mean={mean}, standard deviation={std}"
plt.xlabel("datapoint index")
plt.ylabel("datapoint value")
plt.ylim([-2, 4])
plt.title(title, bbox=style)
plt.savefig(f"images/normal_m_{mean}_std_{std}.pdf")
plt.close()

nbins = 50
plt.hist(x, bins=nbins)
title_hist = "Histogram " + title
plt.xlabel("datapoint value")
plt.ylabel("number of occurrences")
plt.xlim([-1, 3])
plt.title(title, bbox=style)
plt.savefig(f"images/hist_normal_m_{mean}_std_{std}.pdf")
plt.close()
