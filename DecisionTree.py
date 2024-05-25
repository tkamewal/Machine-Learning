# import pandas as pd
# import matplotlib.pyplot as plt

# import numpy as np

# data = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')
# print(data.head())
# x = data.iloc[:, 0:1].values
# y = data.iloc[:, 1].values
# from sklearn.tree import DecisionTreeRegressor
# model = DecisionTreeRegressor()
# model.fit(x, y)
# x_grid = np.arange(min(x), max(x), 0.01)
# x_grid = x_grid.reshape((len(x_grid), 1))
# plt.scatter(x, y)
# plt.plot(x_grid, model.predict(x_grid))
# plt.show()

# Import the necessary libraries
# from sklearn.datasets import load_iris
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import export_graphviz
# from graphviz import Source

# # Load the dataset
# iris = load_iris()
# X = iris.data[:, 2:] # petal length and width
# y = iris.target

# # DecisionTreeClassifier
# tree_clf = DecisionTreeClassifier(criterion='entropy',
# 								max_depth=2)
# tree_clf.fit(X, y)

# # # Plot the decision tree graph
# # export_graphviz(
# # 	tree_clf,
# # 	out_file="iris_tree.dot",
# # 	feature_names=iris.feature_names[2:],
# # 	class_names=iris.target_names,
# # 	rounded=True,
# # 	filled=True
# # )

# # with open("iris_tree.dot") as f:
# # 	dot_graph = f.read()
	
# # Source(dot_graph)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import  mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("C:\\Users\\TANMAY KAMEWAL\\OneDrive\\Desktop\\MachineLearning\\LinearTest.csv")

x = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
regressor = DecisionTreeRegressor(max_depth=3, random_state=1234)
regressor.fit(x_train, y_train)

predictions = regressor.predict([[10]])
print('Prediction for ', predictions)
y_pred = regressor.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE: ", mse)
print("R^2: ", r2)
# print(y_test.ravel())
# print("  ")
# print("  ")
# print("  ")
# print(y_pred)

# x_grid = np.arange(min(x_train), max(x_train), 0.01)
# x_grid = x_grid.reshape((len(x_grid), 1))
# plt.scatter(x_train, y_train)
# plt.plot(x_grid, regressor.predict(x_grid))
# plt.show()

from sklearn import tree

plt.figure(figsize=(15,10))
tree.plot_tree(regressor, filled=True)
plt.show()
