import pandas as pd
from sklearn.datasets import load_iris


X, y = load_iris()

# Load the data to./ a pandas dataframe
df = pd.read_csv("./distribution_3.csv")

# general info on the dataframe
print("---\ngeneral info on the dataframe")
print(df.info())

# print the columns of the dataframe
print("---\ncolumns of the dataset")
print(df.columns)

# print the first 10 lines of the dataframe
print("---\nfirst lines")
print(df.head(10))

# print the correlation matrix of the dataset
print("---\nCorrelation matrix")
print(df.corr())

# print the standard deviation
print("---\nStandard deviation")
print(df.std())
