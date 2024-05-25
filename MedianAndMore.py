import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv('plant.csv')
iris_setosa = iris.loc[iris["Species"] == "Iris-setosa"]
iris_versicolor = iris.loc[iris["Species"] == "Iris-versicolor"]
iris_virginica = iris.loc[iris["Species"] == "Iris-virginica"]

print("Median")
print(np.median(iris_setosa["PetalLength"]))
print(np.median(np.append(iris_setosa["PetalLength"],50)))#outlier
print(np.median(iris_versicolor["PetalLength"]))
print(np.median(iris_virginica["PetalLength"]))

print("Quantiles")
print(np.percentile(iris_setosa["PetalLength"],np.arange(0,100,25)))
print(np.percentile(iris_versicolor["PetalLength"], np.arange(0,100,25)))
print(np.percentile(iris_virginica["PetalLength"], np.arange(0,100,25)))

print("percentile")
print(np.percentile(iris_setosa["PetalLength"],90))
print(np.percentile(iris_versicolor["PetalLength"],90))
print(np.percentile(iris_virginica["PetalLength"],90))
