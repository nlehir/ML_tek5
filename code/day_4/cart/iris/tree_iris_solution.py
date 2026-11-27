from pandas.core.base import doc
from sklearn import tree
from sklearn.datasets import load_iris
import graphviz


def main() -> None:
    """
    Add some lines here, in order to learn a classification tree, with pruning,
    on the iris dataset.

    In order to extract the data from the load_iris() function, we can
    use the relevant attributes, as explained in the docs
    https://scikit-learn.org/1.5/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris

    You can also  experiment with another dataset, like the other scikit toy
    datasets
    https://scikit-learn.org/1.5/datasets/toy_dataset.html
    """
    iris = load_iris()

    X = iris.data
    y = iris.target

    max_depth = 3
    classifier = tree.DecisionTreeClassifier(max_depth=max_depth, criterion="entropy")
    classifier = classifier.fit(X=X, y=y)

    # convert the graph to graphviz in order to visualize it
    dot_data = tree.export_graphviz(
        classifier,
        out_file=None,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render(f"iris_max_depth_{max_depth}")


if __name__ == "__main__":
    main()
