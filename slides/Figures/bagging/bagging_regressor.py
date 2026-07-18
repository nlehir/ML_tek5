# Import the necessary modules and libraries
import math
import numpy as np
import graphviz
# from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.ensemble import BaggingRegressor
import matplotlib.pyplot as plt

xmin = 0
xmax = 2
variance = 0.7
std = math.sqrt(variance)
# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))
nb_samples = 70
X = np.linspace(xmin, xmax, nb_samples).reshape(nb_samples, 1)
noise = rng.normal(0, std, (nb_samples, 1))
modulation = (1-X/xmax)
y = modulation*np.sin(8*X)+noise

# Fit regression model
# max_depth = 3
n_estimators = 100
regr = BaggingRegressor(n_estimators = n_estimators)
regr.fit(X, y.ravel())


# Plot
X_plot = np.arange(xmin, xmax, 0.01)[:, np.newaxis]
y_pred = regr.predict(X_plot)
modulation = (1-X_plot/xmax)
y_bayes = modulation*(np.sin(8*X_plot))


def compute_test_error(n_test):
    X_test = np.random.uniform(xmin, xmax, n_test).reshape(n_test, 1)
    y_pred = regr.predict(X_test)
    modulation = (1-X_test/xmax)
    noise = rng.normal(0, std, (n_test, 1))
    y_truth = modulation*np.sin(8*X_test)+noise
    diff = y_pred-y_truth.ravel()
    test_error = 1/n_test*np.linalg.norm(diff)**2
    return test_error


# tests = list()
# n_tests = [10**i for i in range(1, 7)]
# for n_test in n_tests:
    # tests.append(compute_test_error(n_test))

# plt.plot(np.log10(n_tests), np.log10(tests), "o")
# plt.title("test error")
# plt.savefig("test.pdf")

test_error = compute_test_error(10**6)

# title and params
params = f"\nnumber of estimators: {n_estimators}\n"
params+=f"test error: {test_error:.2E}\n"
params+=f"Bayes risk: {variance:.2E}\n"
title = "Bagging regression"
title+=params

params_fig = f"_nest_{n_estimators}"
figtitle = "Bagging_regression"
figtitle+=params_fig

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_plot, y_pred, color="cornflowerblue", label="prediction", linewidth=2, alpha = 0.8)
plt.plot(X_plot, y_bayes, color="aqua", label="Bayes predictor", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title(title)
plt.legend()
plt.tight_layout()
plt.savefig(figtitle+".pdf")

#indexes = range(3)
#for index in indexes:
#    est = regr.estimators_[index]
#    samples = regr.estimators_samples_[index]
#    title = "samples used\n"+str(samples)
#    dot_data = tree.export_graphviz(est,
#                                    filled=True,
#                                    rounded=True,
#                                    special_characters=True)
#    graph = graphviz.Source(dot_data)
#    graph.render(f"estimator_{index}")
