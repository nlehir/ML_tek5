"""
check test error as a function of the complexity of the tree
"""
import os

import numpy as np
import graphviz
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

fish_features = np.load("data/fish_features_blurred.npy")
fish_class = np.load("data/fish_class_blurred.npy")

# fish_features = np.load("data/fish_features.npy")
# fish_class = np.load("data/fish_class.npy")

feature_names = ["length", "weight", "hour"]
# feature_names = ["length", "weight"]
class_names = ["tuna", "salmon"]

# train_size = 0.7
# test_size = 1-train_size

X_train, X_test, y_train, y_test = train_test_split(fish_features,
                                                    fish_class)
# clean
directory_1 = "images/overfitting/depth"
directory_2 = "images/overfitting/min_samples"
for filename in os.listdir(directory_1):
    os.remove(os.path.join(directory_1, filename))
for filename in os.listdir(directory_2):
    os.remove(os.path.join(directory_2, filename))


def test_depth(max_depth):
    classifier = tree.DecisionTreeClassifier(max_depth=max_depth,
                                             criterion="entropy")
    # train tree on training set
    classifier.fit(X_train, y_train)

    # visualization
    dot_data = tree.export_graphviz(classifier,
                                    out_file=None,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True,
                                    rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render(f"images/overfitting/depth/fish_max_depth_{max_depth}")

    return classifier.score(X_test, y_test)


scores_on_test_set = list()
max_tested = 80
for depth in range(1, max_tested):
    scores_on_test_set.append(test_depth(depth))

plt.plot(range(1, max_tested), scores_on_test_set)
plt.xlabel("max depth")
plt.ylabel("score on test set")
plt.ylim([-0.1, 1.1])
plt.xticks(range(1, max_tested, int(max_tested/10)))
plt.title("influence of tree depth")
plt.savefig("images/overfitting/test error function of depth.pdf")
plt.close()


# test min samples leaf
def test_min_samples_leaf(min_samples_leaf):
    classifier = tree.DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,
                                             criterion="entropy")
    classifier.fit(X_train, y_train)

    # visualization
    dot_data = tree.export_graphviz(classifier,
                                    out_file=None,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True,
                                    rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render(f"images/overfitting/min_samples/fish_min_samples_{min_samples_leaf}")

    return classifier.score(X_test, y_test)


scores_on_test_set = list()
min_tested = 2
max_tested = 80
tested_min_samples_leaf = np.arange(min_tested,
                                    max_tested,
                                    int((max_tested-min_tested)/10))
for depth in tested_min_samples_leaf:
    scores_on_test_set.append(test_min_samples_leaf(depth))

plt.plot(tested_min_samples_leaf, scores_on_test_set)
plt.xlabel("min samples leaf")
plt.ylabel("score on test set")
plt.xticks(tested_min_samples_leaf)
plt.ylim([-0.1, 1.1])
plt.savefig("images/overfitting/est error function of min samples leaf.pdf")
plt.close()
