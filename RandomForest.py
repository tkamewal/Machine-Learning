import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv")

x = data.iloc[:, 0:1].values
y = data.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=300, random_state=0)
model.fit(x_train, y_train)
print(model.predict([[10]]))
print(mean_squared_error(y_test, model.predict(x_test)))
print(r2_score(y_test,model.predict(x_test)))
# x_grid = np.arange(min(x), max(x), 0.01)
# x_grid = x_grid.reshape((len(x_grid), 1))
# plt.scatter(x, y, color="red")
# plt.plot(x_grid, model.predict(x_grid), color="blue")
# plt.show()


