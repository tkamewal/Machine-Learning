# Median Absols ute Deviation (MAD)
import pandas as pd
from statsmodels import robust

iris = pd.read_csv("plant.csv")

iris_setosa = iris.loc[iris.Species == "Iris-setosa"]
iris_versicolor = iris.loc[iris.Species == "Iris-versicolor"]
iris_virginica = iris.loc[iris.Species == "Iris-virginica"]

print(robust.mad(iris_setosa["PetalLength"]))
print(robust.mad(iris_versicolor["PetalLength"]))
print(robust.mad(iris_virginica["PetalLength"]))

