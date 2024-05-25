import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv("Plant.csv")

iris_setosa = iris.loc[iris["Species"] == "Iris-setosa"]

counts, bins = np.histogram(iris_setosa["PetalLength"], bins=10, density=True)

pdf = counts / np.sum(counts)
print(pdf)
print(bins)

cdf = np.cumsum(pdf)
plt.plot(bins[1:], pdf)
plt.plot(bins[1:], cdf)
plt.xlabel("Petal Length")
plt.ylabel("Probability")
# plt.show()

iris_versicolor = iris.loc[iris["Species"] == "Iris-versicolor"]
counts, bins = np.histogram(iris_versicolor["PetalLength"], bins=10, density=True)
pdf = counts / np.sum(counts)
print(pdf)
print(bins)
cdf = np.cumsum(pdf)
plt.plot(bins[1:], pdf)
plt.plot(bins[1:], cdf)
plt.xlabel("Petal Length")
plt.ylabel("Probability")

iris_virginica = iris.loc[iris["Species"] == "Iris-virginica"]
counts, bins = np.histogram(iris_virginica["PetalLength"], bins=10, density=True)
pdf = counts / np.sum(counts)
print(pdf)
print(bins)
cdf = np.cumsum(pdf)
plt.plot(bins[1:], pdf)
plt.plot(bins[1:], cdf)
plt.xlabel("Petal Length setosa versicolor verginica")
plt.ylabel("Probability")
plt.scatter(5,0.95)
plt.show()


