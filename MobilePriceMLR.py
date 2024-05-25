import pandas as pd
import numpy as np

data = pd.read_csv("Mobile.csv")
print(data.head())

x = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values
y = y * 10000
# Now we need to convert categorical data into numerical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

le = LabelEncoder()
x[:, 1] = le.fit_transform(x[:, 1])

# Use ColumnTransformer for one-hot encoding
column_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), [1])  # Specify the column index to one-hot encode
    ],
    remainder='passthrough'  # Keep the remaining columns as they are
)

x = column_transformer.fit_transform(x)

# Avoiding Dummy Variable Trap
x = x[:, 1:]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print(y_test[:10])

Round_y_pred = np.round(y_pred)
print(Round_y_pred[:10])

# backward elimination
import statsmodels.regression.linear_model as sm
x = np.append(arr=np.ones((x.shape[0], 1)).astype(int), values=x, axis=1)
regressor_ols = sm.OLS(endog=y, exog=x).fit()

# Backward elimination loop
while True:
    # Check p-values
    p_values = regressor_ols.pvalues[1:]
    max_p_value = max(p_values)

    # Remove the feature with the highest p-value if greater than the threshold
    if max_p_value > 0.05:
        max_p_index = np.argmax(p_values)
        x = np.delete(x, max_p_index + 1, axis=1)  # Remove corresponding feature
        regressor_ols = sm.OLS(endog=y, exog=x).fit()
    else:
        break  # Exit the loop if all p-values are below the threshold

# Display the final summary
# print(regressor_ols.summary())

# Extract automatically selected features
selected_feature_indices = np.where(regressor_ols.pvalues[1:] <= 0.05)[0]
x_selected_auto = x[:, selected_feature_indices + 1]  # Adding 1 to account for the constant term

# Train-test split for the updated dataset with selected features
x_train_auto, x_test_auto, y_train_auto, y_test_auto = train_test_split(x_selected_auto, y, test_size=0.2, random_state=0)

# Train a multiple linear regression model with automatically selected features
regressor_selected_auto = LinearRegression()
regressor_selected_auto.fit(x_train_auto, y_train_auto)

# Print results for the multiple linear regression with automatically selected features
# print("\nMultiple Linear Regression with Automatically Selected Features:")
# print("Selected Feature Indices:", selected_feature_indices)
# print("Coefficients:", regressor_selected_auto.coef_)
# print("Intercept:", regressor_selected_auto.intercept_)
# print("R-squared:", regressor_selected_auto.score(x_test_auto, y_test_auto))
# print(regressor_selected_auto.predict(x_test_auto))
y_pred1 = regressor_selected_auto.predict(x_test_auto)
print(y_test_auto[:10])
Round_y_pred1 = np.round(y_pred1)
print(Round_y_pred1[:10])
