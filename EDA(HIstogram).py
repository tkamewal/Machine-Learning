import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv("plant.csv")

iris_setosa = iris.loc[iris["Species"] == "Iris-setosa"]
iris_versicolor = iris.loc[iris["Species"] == "Iris-versicolor"]
iris_virginica = iris.loc[iris["Species"] == "Iris-virginica"]

plt.plot(iris_setosa["PetalLength"], np.zeros_like(iris_setosa["PetalLength"]), "o")
plt.plot(iris_versicolor["PetalLength"], np.zeros_like(iris_versicolor["PetalLength"]), "o")
plt.plot(iris_virginica["PetalLength"], np.zeros_like(iris_virginica["PetalLength"]), "o")
# plt.show()

sns.FacetGrid(iris, hue="Species", height=5).map(sns.distplot, "PetalLength").add_legend()
plt.show()