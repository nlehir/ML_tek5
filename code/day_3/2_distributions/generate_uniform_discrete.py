import matplotlib.pyplot as plt
import numpy as np

nb_data = 400
nb_values = 10

style = {"facecolor": "blue", "alpha": 0.2, "pad": 10}

x = np.random.randint(0, nb_values, nb_data)
plt.plot(range(nb_data), x, "o")
title = "Unifom discrete distribution"
plt.xlabel("datapoint index")
plt.ylabel("datapoint value")
plt.title(title, bbox=style)
plt.savefig(f"images/uniform_discrete_{nb_values}.pdf")
plt.close()

nbins = 50
plt.hist(x, bins=nbins)
title_hist = "Histogram " + title
plt.xlabel("datapoint value")
plt.ylabel("number of occurrences")
plt.title(title, bbox=style)
plt.savefig(f"images/hist_uniform_discrete_{nb_values}.pdf")
plt.close()
