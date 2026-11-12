import matplotlib.pyplot as plt
import numpy as np

nb_data = 50000
x_min = 0
x_max = 1

style = {"facecolor": "blue", "alpha": 0.2, "pad": 10}
x = np.random.uniform(x_min, x_max, nb_data)
plt.plot(range(nb_data), x, "o", markersize=1)
title = f"Uniform distribution : [{x_min}, {x_max}]"
plt.xlabel("datapoint index")
plt.ylabel("datapoint value")
plt.title(title, bbox=style)
# plt.ylim([34, 55])
plt.savefig(f"images/unif_continuous_{x_min}_{x_max}.pdf")
plt.close()

nbins = 10
plt.hist(x, bins=nbins)
title_hist = "Histogram " + title
plt.xlabel("datapoint value")
plt.ylabel("number of occurrences")
plt.title(title, bbox=style)
plt.savefig(f"images/hist_unif_continuous_{x_min}_{x_max}.pdf")
plt.close()
