import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Position_Salaries.csv')
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
print("X:\n", x)
print("\nY:\n", y)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)
print("X_Poly", x_poly)
print(lin_reg.predict(poly.fit_transform([[6.5]])))
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
