import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv")
print(data.head())
x = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, 1].values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
plt.scatter(x_train, y_train,color='r',marker='o')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience")
plt.show()
model = LinearRegression()
model.fit(x_train, y_train)
# print(model.coef_)
# print(model.intercept_)
# print(model.score(x_test, y_test))
y_pred = model.predict(x_test)
# print(y_test)
# print(y_pred)
plt.scatter(x_train, y_train, color='g', marker='o')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience")
plt.plot(x_train,model.predict(x_train))
plt.show()
print(model.predict([[10]]))

mse = mean_squared_error(y_test, y_pred)
print(mse)



