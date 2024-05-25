import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv('plant.csv')
iris_setosa = iris.loc[iris["Species"] == "Iris-setosa"]
iris_versicolor = iris.loc[iris["Species"] == "Iris-versicolor"]
iris_virginica = iris.loc[iris["Species"] == "Iris-virginica"]
 
print("Mean")
print(np.mean(iris_setosa["PetalLength"]))
print(np.mean(np.append(iris_setosa["PetalLength"],50)))#outlier affect mean
print(np.mean(iris_versicolor["PetalLength"]))
print(np.mean(iris_virginica["PetalLength"]))

print("STD")
print(np.std(iris_setosa["PetalLength"]))
print(np.std(iris_versicolor["PetalLength"]))
print(np.std(iris_virginica["PetalLength"]))