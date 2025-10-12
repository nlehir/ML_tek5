"""
build a cart for the fish dataset
"""

import numpy as np
from sklearn import tree
import graphviz

def predict_new_fish(classifier,
                     new_fish_length,
                     new_fish_weight,
                     ) -> None:
    new_fish_1_array = np.array([[new_fish_length, new_fish_weight]])
    prediction = classifier.predict(new_fish_1_array)
    if prediction == 1:
        predicted_class = "salmon"
    else:
        predicted_class = "tuna"
    print(f"\nnew fish: {new_fish_length} centimeters, {new_fish_weight} kilos")
    print(f"the class predicted for the new fish is {predicted_class}")


def main() -> None:
    fish_features = np.load("data/fish_features.npy")
    fish_class = np.load("data/fish_class.npy")

    feature_names = ["length", "weight"]
    class_names = ["tuna", "salmon"]

    max_depth = 100
    classifier = tree.DecisionTreeClassifier(max_depth=max_depth, criterion="entropy")
    classifier = classifier.fit(fish_features, fish_class)

    # convert the graph to graphviz in order to visualize it
    dot_data= tree.export_graphviz(classifier,
                                   out_file=None,
                                   feature_names=feature_names,
                                   class_names=class_names,
                                   filled=True,
                                   rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render(f"images/trees/fish_max_depth_{max_depth}")

    # use the tree for prediction
    predict_new_fish(
            classifier=classifier,
            new_fish_length = 35,
            new_fish_weight = 4,
            )
    predict_new_fish(
            classifier=classifier,
            new_fish_length = 60,
            new_fish_weight = 7,
            )

if __name__ == "__main__":
    main()
