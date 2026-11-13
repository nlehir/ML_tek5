"""
Recursive feature elimination (RFE) with logistic regression as estimator
"""

import os

import numpy as np
import utils
from constants import MIN_DF, NGRAM_RANGE, NUM_JOBS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from utils_data_processing import preprocess_imdb

C = 0.5
FINAL_NB_FEATURES = 10000
FEATURES_REMOVED_PER_ITERATION = 10000


if __name__ == "__main__":
    cache_name = "imdb_wrapper.pkz"
    traindata, _, testdata = preprocess_imdb(num_jobs=NUM_JOBS)

    print("Vectorizing the data")
    vectorizer = CountVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF)
    X_train = vectorizer.fit_transform(traindata.data)
    y_train = traindata.target
    X_test = vectorizer.transform(testdata.data)
    y_test = testdata.target
    vocabulary = np.array(vectorizer.get_feature_names_out())

    print(f"Original vocabulary size : {len(vocabulary)}")

    classifier = Pipeline(
        [("scaler", MaxAbsScaler()), ("clf", LogisticRegression(solver="liblinear"))]
    )
    selector = RFE(
        classifier,
        n_features_to_select=FINAL_NB_FEATURES,
        step=FEATURES_REMOVED_PER_ITERATION,
        importance_getter="named_steps.clf.coef_",
        verbose=1,
    )
    print("Performing the recursive feature elimination")
    selector.fit(X_train, y_train)

    acc_train = selector.score(X_train, y_train)
    acc_test = selector.score(X_test, y_test)
    scores = cross_val_score(
        selector, X_train, y_train, cv=5, n_jobs=NUM_JOBS, verbose=1
    )

    print(f"Cross validated accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
    print(f"train accuracy : {acc_train:.2f}")
    print(f"test accuracy : {acc_test:.2f}")

    selected_dims = selector.get_support()
    selected_terms = vocabulary[selected_dims]
    weights = selector.estimator_.named_steps["clf"].coef_.ravel()
    sorted_idx = np.argsort(weights)

    print(f"Original vocabulary size : {len(vocabulary)}")
    print(f"Selected vocabulary size : {len(weights)}")

    file_name = "vocabulary_exercise_4_wrapper.txt"
    print(f"save vocabulary to {file_name}")
    file_path = os.path.join("vocabularies", file_name)
    with open(file_path, "w") as out_file:
        out_file.write(
            "\n".join(
                [
                    f"{word} ({weight})"
                    for word, weight in zip(
                        selected_terms[sorted_idx], weights[sorted_idx]
                    )
                ]
            )
        )

