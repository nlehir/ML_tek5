import os
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


sample_indexes = [1, 2, 3, 12, 100]
for sample_index in sample_indexes:
    digits = load_digits()
    targets = digits.target
    plt.imshow(digits.data[sample_index].reshape(8, 8))
    title = f"label: {targets[sample_index]}"
    plt.title(title)
    figpath = os.path.join("images", f"sample_{sample_index}.pdf")
    plt.savefig(figpath)
