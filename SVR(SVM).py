import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("C:\\Users\\TANMAY KAMEWAL\\OneDrive\\Desktop\\MachineLearning\\LinearTest.csv")
# data.head()
# x = data.iloc[:, :-1].values
# # print(x)
# y = data.iloc[:, -1].values
# # print(y)

# # Define the parameter grid
# param_grid = {
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
# }

# # Initialize SVR
# svr = SVR()

# # Initialize GridSearchCV
# grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=3)

# # Fit GridSearchCV to data
# grid_search.fit(x, y)

# # Print the best kernel
# print("Best kernel:", grid_search.best_params_['kernel'])

# best_svr = grid_search.best_estimator_
# print(best_svr)

#  # Calculate the standard deviation of the input feature
# std_deviation_x = x.std()

#         # Calculate the standard deviation of the output feature
# std_deviation_y = y.std()

#         # Set a threshold as a multiple of the standard deviation
# threshold_multiplier = 2  # You can adjust this multiplier based on your data and requirements

#         # Check if standard scaling is required based on the standard deviation
# if std_deviation_x > threshold_multiplier:
#     scaler = StandardScaler()
#     use_scaling_x = True
# else:
#     use_scaling_x = False

#         # Check if standard scaling is required based on the standard deviation
# if std_deviation_y > threshold_multiplier:
#     scaler1 = StandardScaler()
#     use_scaling_y = True
# else:
#     use_scaling_y = False


# if use_scaling_x:
#     scaler.fit_transform(x)

# if use_scaling_y:
#     y = scaler1.fit_transform(y.reshape(-1, 1))

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# from sklearn.svm import SVR

# svr = SVR(kernel='linear', C=1)
# svr.fit(x, y.ravel())
# # svr.predict(x_test)
# # plt.scatter(x, y)

# # plt.plot(x, svr.predict(x), color="red")
# # plt.show()
# # y_pred =  svr.predict(x_test).reshape(-1, 1)
# # print("before...")
# # print(y_test.reshape(-1))
# predict = np.array([[10, 10000]])
# prediction = scaler.transform(predict)
# prediction_values = scaler1.inverse_transform(svr.predict(prediction))
# # r2 = r2_score(y_test, y_pred)
# print("Predicted value: ", prediction_values)
# # print("R^2 score: ", r2)
# print(" ")
# # print(scaler1.inverse_transform(y_test).reshape(-1, 1))
# print(scaler1.inverse_transform(y_pred).reshape(-1, 1))

# Obtain cross-validated predictions
# y_pred = cross_val_predict(svr, X, y, cv=5)

# Scatter plot with SVR predictions
# plt.figure(figsize=(8, 6))
# plt.scatter(X, y, color='blue', label='Actual')
# plt.scatter(X, y_pred, color='red', label='Predicted')
# plt.plot(X, y_pred, color='green', linewidth=2)
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('SVR: Actual vs Predicted')
# plt.legend()
# plt.show()

# Residual plot

# # Calculate residuals correctly
# residuals = y.squeeze() - y_pred.squeeze()
# print("Shape of y:", y.shape)
# print("Shape of residuals:", residuals.shape)
# print("residuals:", residuals)

# plt.figure(figsize=(8, 6))
# plt.scatter(X, residuals, color='blue')
# plt.axhline(y=0, color='red', linestyle='--')
# plt.xlabel('X')
# plt.ylabel('Residuals')
# plt.title('Residual Plot')
# plt.show()

# # Density plot of residuals
# plt.figure(figsize=(8, 6))
# plt.hist(residuals, bins=20, density=True, color='skyblue', edgecolor='black', alpha=0.7)
# plt.xlabel('Residuals')
# plt.ylabel('Density')
# plt.title('Density Plot of Residuals')
# plt.show()

# Learning curve (MSE)
# train_errors, test_errors = [], []
# for m in range(1, len(X)):
#     svr.fit(X[:m], y[:m])
#     y_train_pred = svr.predict(X[:m])
#     y_test_pred = svr.predict(X)
#     train_errors.append(mean_squared_error(y[:m], y_train_pred))
#     test_errors.append(mean_squared_error(y, y_test_pred))

# plt.figure(figsize=(8, 6))
# plt.plot(np.sqrt(train_errors), 'r-+', linewidth=2, label='train')
# plt.plot(np.sqrt(test_errors), 'b-', linewidth=3, label='test')
# plt.xlabel('Training set size')
# plt.ylabel('RMSE')
# plt.title('Learning Curve')
# plt.legend()
# plt.show()

# from sklearn import svm
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import numpy as np

# # Splitting data into features and target variable
# X = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values

# # Standard Scaling
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()

# X_scaled = scaler_X.fit_transform(X)
# y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# # Splitting the dataset into the Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25, random_state=42)

# # Defining the SVR model
# svr = SVR()

# # Parameter grid for GridSearchCV
# param_grid = {
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'C': [0.1, 1, 10, 100],
#     'gamma': ['scale', 'auto'],
#     'epsilon': [0.1, 0.2, 0.5]
# }

# # Performing GridSearchCV
# grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)

# # Getting the best parameters
# best_params = grid_search.best_params_
# print("Best Parameters:", best_params)

# # Predicting the Test set results
# y_pred = grid_search.predict(X_test)

# # Inverse transforming the predictions to get actual values
# y_pred_actual = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
# # print(y_pred_actual)

# # Performing prediction for specified input [10, 10000]
# future_input = np.array([[10, 10000]])
# future_input_scaled = scaler_X.transform(future_input)
# prediction_scaled = grid_search.predict(future_input_scaled)
# prediction_actual = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()
# print("Prediction for [10, 10000]:", prediction_actual)
# r2_score = r2_score(y_test, y_pred)
# print(r2_score)



# from sklearn.preprocessing import StandardScaler

# x = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values

# # print(x[:5])
# # print(y[:5])

# scaler_X = StandardScaler()
# scaler_y = StandardScaler()

# X_scaled = scaler_X.fit_transform(x)

# y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
# x = scaler_X.inverse_transform(X_scaled)
# y = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
# # print(x[:5])
# # print(y[:5])

# scaler_y.inverse_transform(svr.predict(x[:, 0].reshape(-1, 1))).reshape(-1, 1)

 # # Get support vectors
    # support_vectors = svr.support_vectors_
    # # Plot the data, SVR predictions, and support vectors
    # plt.figure(figsize=(10, 6))
    # plt.scatter(x_train, y_train, color='blue', label='Train Data')
    # plt.scatter(x_test, y_test, color='green', label='Test Data')
    # plt.plot(np.concatenate([x_train, x_test]), svr.predict(np.concatenate([x_train, x_test])), color='red',
    #          label='SVR Predictions')
    # plt.scatter(support_vectors, svr.predict(support_vectors), color='orange', label='Support Vectors')
    # plt.scatter(predict_value, prediction_results, color='black', marker='o', label='Predicted Value')
    # plt.xlabel('X')
    # plt.ylabel('y')
    # plt.title('Support Vector Regression with Margin Lines')
    # plt.legend()
    #
    # # Plotting margin lines
    # plt.plot(np.concatenate([x_train, x_test]), svr.predict(np.concatenate([x_train, x_test])), color='black',
    #          linestyle='-.', alpha=0.3)
    # plt.plot(np.concatenate([x_train, x_test]), svr.predict(np.concatenate([x_train, x_test])) + svr.epsilon,
    #          color='black', linestyle='-.', alpha=0.3)
    # plt.plot(np.concatenate([x_train, x_test]), svr.predict(np.concatenate([x_train, x_test])) - svr.epsilon,
    #          color='black', linestyle='-.', alpha=0.3)
    #
x = data.iloc[:,:-1].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
# sc_x = StandardScaler()
# sc_y = StandardScaler()

# x_train = sc_x.fit_transform(x_train)
# y_train = sc_y.fit_transform(y_train.reshape(-1,1))


# Defining the SVR model
svr = SVR()
# svr.fit(x_train, y_train)

# Parameter grid for GridSearchCV
param_grid = {
            'kernel': [ 'poly', 'rbf', 'sigmoid','linear'],
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'epsilon': [0.1, 0.2, 0.5]
}

# Performing GridSearchCV
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)  # Note: Use fit() instead of just declaring grid_search

# # # Get the best parameters found
best_params = grid_search.best_params_

# # # Train the SVR model with the best parameters
best_svr = SVR(**best_params)
best_svr.fit(x_train, y_train)
# # print("Best SVR model:", best_svr)
# # print("Best parameters:", best_params)
# # # y_pred = best_svr.predict(x_test)
# # y_val = np.array([[20]])
# # y_val_scaled = sc_x.transform(y_val)
# # print("Y-val_scaled=",y_val_scaled)
# # predict = best_svr.predict(y_val_scaled)
# # print("Predict=",predict)
# # predict = sc_y.inverse_transform(predict.reshape(-1, 1)).ravel()
# # print(predict)
# # svr = SVR(kernel='linear')
# # svr.fit(x_train, y_train)
# plo = grid_search.predict(x_train)
# # plo = sc_y.inverse_transform(plo.reshape(-1,1)).ravel()
# # print('PLO= ', plo)
# y_pred= best_svr.predict(x_test)
# epsilon = best_svr.epsilon
# # plt.scatter(x_train, y_train, color='darkorange', label='data')
# # plt.plot(x_test, y_test, color='pink', lw=2, label='RBF model')
# # Compute upper and lower boundary lines
# upper_boundary = y_pred + epsilon
# lower_boundary = y_pred - epsilon

# # Plotting
# # plt.scatter(x, y, color='darkorange', label='data')
# # plt.plot(x_test, y_pred, color='navy', lw=2, label='RBF model')
# plt.plot(x_test, upper_boundary, 'k--')
# plt.plot(x_test, lower_boundary, 'k--')
# plt.fill_between(x_test[:, 0], upper_boundary, lower_boundary, color='lightgray', alpha=0.5)
# plt.xlabel('data')
# plt.ylabel('target')
# plt.title('Support Vector Regression')
# plt.legend()
# plt.show()


# min_x = np.min(x_train) - 1
# max_x = np.max(x_train) + 1
# x_visualization = np.linspace(min_x, max_x, 100).reshape((-1, 1))

# # Perform the prediction on the visualization data
# plo = grid_search.predict(x_visualization)

# # Calculate the margin boundaries
# boundary_offset = best_svr.dual_coef_ * (x_train.mean(axis=0) - x_train.min(axis=0))
# boundary_offset = np.squeeze(boundary_offset)  # Remove any unnecessary dimensions
# margin_boundary = np.zeros((x_visualization.shape[0], 2))

# # Ensure that both plo and boundary_offset have the same shape
# if len(plo.shape) == 1:
#     plo = plo.reshape(-1, 1)

# # Calculate the margin boundaries
# num_features = x_visualization.shape[1]
# mean_diff = x_train.mean(axis=0) - x_train.min(axis=0)
# # Reshape best_svr.dual_coef_ to (n_support_vectors, 1)
# dual_coef_reshaped = best_svr.dual_coef_.reshape(-1, 1)

# # Calculate the margin boundaries
# boundary_offset = np.dot(dual_coef_, mean_diff.reshape(-1, 1)).ravel()
# boundary_offset = np.repeat(boundary_offset.reshape(1, -1), x_visualization.shape[0], axis=0)

# # Ensure that both plo and boundary_offset have the same shape
# if plo.shape != boundary_offset.shape:
#     raise ValueError("Shapes of plo and boundary_offset are not compatible.")



# margin_boundary[:, 0] = plo.flatten() + best_svr.epsilon - boundary_offset.flatten()
# margin_boundary[:, 1] = plo.flatten() - best_svr.epsilon + boundary_offset.flatten()



# # Plot the data
# plt.scatter(x_train, y_train, color='black', label='Training data points')
# plt.plot(x_visualization, plo, color='blue', label='Regression line')
# plt.plot(x_visualization[margin_boundary[:, 0] < 0, :], plo[margin_boundary[:, 0] < 0], color='red', linestyle='--', label='Lower margin boundary')
# plt.plot(x_visualization[margin_boundary[:, 1] > 0, :], plo[margin_boundary[:, 1] > 0], color='red', linestyle='--', label='Upper margin boundary')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.show()

# # # Perform the prediction on the test data
# y_pred = best_svr.predict(x_test)

# # # Print the results
# # print('Best SVR model:', best_svr)
# # print('Best parameters:', best_params)
# # print('Predictions:', y_pred)


# # # Plot the results
# # plt.scatter(x_train, y_train, color='darkorange', label='data')
# # plt.plot(x_test, y_pred, color='navy', lw=2, label='RBF model')

# # # Plot support vectors
# # plt.scatter(x_train[best_svr.support_], y_train[best_svr.support_], facecolors='none', edgecolors='k', label='support vectors')

# # # Plot margin boundaries
# # plt.plot(x_test, y_pred + best_svr.epsilon, 'k--')
# # plt.plot(x_test, y_pred - best_svr.epsilon, 'k--')

# # plt.xlabel('data')
# # plt.ylabel('target')
# # plt.title('Support Vector Regression')
# # plt.legend()
# # plt.show()

# epsilon = best_svr.epsilon
# print(epsilon)

# # plt.plot(x_test, y_test + epsilon, 'k--')
# # plt.plot(x_test, y_test - epsilon, 'k--')
# # plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Example input and output data
input_data = x.ravel()
output_data = y

# Fit a linear regression line
slope, intercept = np.polyfit(input_data, output_data, 1)
fit_line = slope * input_data + intercept

# Plot the input and output data points
plt.scatter(input_data, output_data, color='blue', label='Data Points')

# Plot the regression line
plt.plot(input_data, fit_line, color='red', label='Regression Line')

# Add labels and legend
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Linear Regression')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
