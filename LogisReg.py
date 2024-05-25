import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

# Load the dataset
data = pd.read_csv("C:\\Users\\TANMAY KAMEWAL\\OneDrive\\Desktop\\MachineLearning\\Social_Network_Ads.csv")

# Extract features and target variable
X = data.iloc[:, :-1].values  # Features: Age and EstimatedSalary
y = data.iloc[:, -1].values       # Target variable: Purchased

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Fit logistic regression model to the training data
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

# Predict the test set results
y_pred = model.predict(X_test)

# Confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(cm)
# acc = accuracy_score(y_test, y_pred) *  100
# print(acc)
# cf  = classification_report(y_test, y_pred)
# print(cf)
# f1 = f1_score(y_test, y_pred)
# print("F1 Score ", f1)

# Visualizing the training set results
# plt.figure(figsize=(10, 5))

# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
#                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
# plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              color=('red, green'), alpha=0.75)
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for  i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j,  1],
#                  label=j)
#     plt.title('Logistic Regression (Training set)')
#     plt.xlabel('Age')
#     plt.ylabel('Estimated Salary')
#     plt.legend()
# plt.show()
# # Visualizing the test set results
# #plt.figure(figsize=(10, 5))

# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
#                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
# plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha=0.75)

# plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set)
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# plt.title('Logistic Regression (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import numpy as np

# # 1. Confusion Matrix
y_pred = model.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(cm)

# # # 2. Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# # 3. Precision
# precision = precision_score(y_test, y_pred, average="binary")
# print("Precision:", precision)

# # 4. Recall
# recall = recall_score(y_test, y_pred)
# print("Recall:", recall)

# 5. F1 Score
# f1 = f1_score(y_test, y_pred)
# print("F1 Score:", f1)
# print(y_test)
# print(" ")
# print(y_pred)

# # 6. ROC Curve and AUC Score
y_pred_proba = model.predict_proba(X_test)[:, 1]
# y_pred_proba1 = model.predict_proba(X_test)[:, 0]
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred_proba1)
# roc_auc = roc_auc_score(y_test, y_pred_proba)
# roc_auc1 = roc_auc_score(y_test, y_pred_proba1)
# print("AUC Score:", roc_auc)

# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc )
# plt.plot(fpr1, tpr1, label='ROC curve (area = %0.2f)' % roc_auc1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc="lower right")
# plt.show()

# # 7. Log-loss (Cross-Entropy Loss)
# logloss = log_loss(y_test, y_pred_proba)
# print("Log-loss:", logloss)

# # 8. Calibration Curve
# prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy='uniform')

# plt.plot(prob_pred, prob_true, marker='o', linestyle='-')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('Mean predicted probability')
# plt.ylabel('Fraction of positives')
# plt.title('Calibration Curve')
# plt.show()

# # # 9. Feature Importance (if applicable)
# if hasattr(model, 'coef_'):
#     feature_importance = np.abs(model.coef_[0])
#     print("Feature Importance:")
#     print(feature_importance)

# # 10. Cross-validation (if applicable)
# predicted = cross_val_predict(model, X, y, cv=5)
# # Use the predicted values for further analysis or evaluation

# print(data["Purchased"].nunique())
# a=data["y"].nunique()
# print(a)

# prediction = model.predict([[25, 25000]])
# print("Predicted Class Probabilities: ", prediction)

# cm 
# acc 
# cf 
# f1 
# precision
# recall
# roc_auc
# logloss
# feature_importance


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import make_classification
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# # Generate some example data
# X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, random_state=42)

# # Fit logistic regression model
# clf = LogisticRegression()
# clf.fit(X, y)

# # Generate predictions
# y_pred = clf.predict(X)

# # Compute confusion matrix
# cm = confusion_matrix(y, y_pred)

# # Plot confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.special import expit

# from sklearn.linear_model import LinearRegression, LogisticRegression

# # Generate a toy dataset, it's just a straight line with some Gaussian noise:
# xmin, xmax = -5, 5
# n_samples = 100
# np.random.seed(0)
# X = np.random.normal(size=n_samples)
# y = (X > 0).astype(float)
# X[X > 0] *= 4
# X += 0.3 * np.random.normal(size=n_samples)

# X = X[:, np.newaxis]

# # Fit the classifier
# clf = LogisticRegression(C=1e5)
# clf.fit(X, y)

# # and plot the result
# plt.figure(1, figsize=(4, 3))
# plt.clf()
# plt.scatter(X.ravel(), y, label="example data", color="black", zorder=20)
# X_test = np.linspace(-5, 10, 300)

# loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
# plt.plot(X_test, loss, label="Logistic Regression Model", color="red", linewidth=3)

# ols = LinearRegression()
# ols.fit(X, y)
# plt.plot(
#     X_test,
#     ols.coef_ * X_test + ols.intercept_,
#     label="Linear Regression Model",
#     linewidth=1,
# )
# plt.axhline(0.5, color=".5")

# plt.ylabel("y")
# plt.xlabel("X")
# plt.xticks(range(-5, 10))
# plt.yticks([0, 0.5, 1])
# plt.ylim(-0.25, 1.25)
# plt.xlim(-4, 10)
# plt.legend(
#     loc="lower right",
#     fontsize="small",
# )
# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def logistic_regression(x_train, w, b):
#     return sigmoid(np.dot(x_train, w) + b)

# # Generate some sample data
# np.random.seed(0)
# num_samples = 300
# x_train = np.random.randn(num_samples, 1)
# y_train = np.random.randint(0, 2, size=(num_samples,))

# # Initialize weights and bias
# w = np.random.randn(1)
# b = np.random.randn()

# # Train logistic regression using gradient descent
# learning_rate = 0.01
# num_epochs = 1000
# for epoch in range(num_epochs):
#     # Calculate predictions
#     predictions = logistic_regression(x_train, w, b)

#     # Calculate gradients
#     dw = np.dot(x_train.T, (predictions - y_train))
#     db = np.sum(predictions - y_train)

#     # Update weights and bias
#     w -= learning_rate * dw
#     b -= learning_rate * db

# # Plot the data and the logistic regression curve
# plt.scatter(x_train, y_train, color='blue', label='Data points')
# plt.xlabel('x')
# plt.ylabel('y')

# x_values = np.linspace(-3, 3, 100)
# plt.plot(x_values, logistic_regression(x_values[:, np.newaxis], w, b), color='red', label='Logistic Regression')

# plt.title('Logistic Regression with Sigmoid Function')
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(1, figsize=(5, 5))
# plt.clf()
# plt.scatter(X_train[:, 0].ravel(), y_train, label="example data", color="black", zorder=20)
# # Generate test data based on the training data distribution
# x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
# y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                      np.arange(y_min, y_max, 0.1))
# # X_test = np.c_[xx.ravel(), yy.ravel()]

# # Compute decision function for logistic regression
# decision_function = model.decision_function(X_test)
# from scipy.special import  expit
# # Compute probability estimates for logistic regression
# loss = expit(decision_function)

# plt.plot(X_test, loss, label="Logistic Regression Model", color="skyblue", linewidth=3)
# from sklearn.linear_model import LinearRegression
# ols = LinearRegression()
# ols.fit(X_train, y_train)
# plt.plot(
#     X_test,
#     ols.coef_ * X_test + ols.intercept_,
#     label="Linear Regression Model",
#     linewidth=1,
# )
# plt.axhline(0.5, color=".5")

# plt.ylabel("y")
# plt.xlabel("X")
# plt.xticks(range(-5, 10))
# plt.yticks([0, 0.5, 1])
# plt.ylim(-0.25, 1.25)
# plt.xlim(-4, 10)
# plt.legend(
#     loc="lower right",
#     fontsize="small",
# )
# plt.tight_layout()
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.special import expit

# from sklearn.linear_model import LinearRegression, LogisticRegression

# # Generate a toy dataset, it's just a straight line with some Gaussian noise:
# xmin, xmax = -5, 5
# n_samples = 100
# np.random.seed(0)
# X = np.random.normal(size=n_samples)
# y = (X > 0).astype(float)
# X[X > 0] *= 4
# X += 0.3 * np.random.normal(size=n_samples)

# X = X[:, np.newaxis]

# # Fit the classifier
# clf = LogisticRegression(C=1e5)
# clf.fit(X, y)

# # and plot the result
# plt.figure(1, figsize=(4, 3))
# plt.clf()
# plt.scatter(X.ravel(), y, label="example data", color="black", zorder=20)
# X_test = np.linspace(-5, 10, 300)
# print(clf.coef_.shape)
# print(model.coef_.shape)
# print(X_test.shape)
# loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
# plt.plot(X_test, loss, label="Logistic Regression Model", color="red", linewidth=3)
# print(loss.shape)
# ols = LinearRegression()
# ols.fit(X, y)
# plt.plot(
#     X_test,
#     ols.coef_ * X_test + ols.intercept_,
#     label="Linear Regression Model",
#     linewidth=1,
# )
# plt.axhline(0.5, color=".5")

# plt.ylabel("y")
# plt.xlabel("X")
# plt.xticks(range(-5, 10))
# plt.yticks([0, 0.5, 1])
# plt.ylim(-0.25, 1.25)
# plt.xlim(-4, 10)
# plt.legend(
#     loc="lower right",
#     fontsize="small",
# )
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from sklearn.linear_model import LinearRegression, LogisticRegression

# Generate a toy dataset, it's just a straight line with some Gaussian noise:
# xmin, xmax = -5, 5
# n_samples = 100
# np.random.seed(0)
# X = np.random.normal(size=n_samples)
# y = (X > 0).astype(float)
# X[X > 0] *= 4
# X += 0.3 * np.random.normal(size=n_samples)

# X = X[:, np.newaxis]

# Fit the classifier
# clf = LogisticRegression(C=1e5)
# clf.fit(X_train, y_train)

# # and plot the result
# plt.figure(1, figsize=(4, 3))
# plt.clf()
# plt.scatter(X_train[:, 0].ravel(), y_train, label="example data", color="black", zorder=20)
# X_test = np.linspace(-5, 10, 300).reshape(-1, 1)  # Reshape to have two dimensions

# # Calculate Z for logistic regression model
# Z = X_test * clf.coef_[0][0] + X_test * clf.coef_[0][1] + clf.intercept_
# loss = expit(Z).ravel()
# plt.plot(X_test, loss, label="Logistic Regression Model", color="red", linewidth=3)

# import seaborn as sns
# sns.regplot(
#     x=X_train[:, 0], y=model.predict(X_train),
#     line_kws={'color': 'red', 'alpha': 0.7, 'linestyle': '--', 'linewidth': 2, 'label': 'Regression Line'},
# )
# plt.axhline(0.5, color=".5")


# plt.ylabel("y")
# plt.xlabel("X")
# plt.xticks(range(-5, 10))
# plt.yticks([0, 0.5, 1])
# plt.ylim(-0.25, 1.25)
# plt.xlim(-4, 10)
# plt.legend(loc="lower right", fontsize="small")
# plt.tight_layout()
# plt.show()
# predict = sc.transform(np.array([[27, 35000]]))
# print(predict)

# print(model.predict(predict))
# import seaborn as sns
# sns.regplot(x=X[:, 1], y=y, data=data, logistic=True, ci=None)
# plt.title('Logistic Regression')
# # Calculate Z for logistic regression model
# Z = X_test * model.coef_[0][0] + X_test * model.coef_[0][1] + model.intercept_
# loss = expit(Z).ravel()
# # plt.plot(X_test, loss, label="Logistic Regression Model", color="red", linewidth=3)

# plt.axhline(0.5, color=".5")

# ols = LinearRegression()
# ols.fit(X, y)
# plt.plot(
#             X_test,
#             ols.coef_ * X_test + ols.intercept_,
#             label=f"Linear Regression Model",
#             linewidth=1,
#         )
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Generate synthetic data
# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
#                            n_clusters_per_class=1, random_state=42)
# X= sc.fit_transform(X)
# # Fit logistic regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Generate grid points
# x1_min, x1_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
# x2_min, x2_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
# xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
#                        np.arange(x2_min, x2_max, 0.02))

# # Predict class probabilities for grid points
# Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
# Z = Z.reshape(xx1.shape)

# # Plot decision boundary and data points
# plt.contourf(xx1, xx2, Z, alpha=0.8)
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Logistic Regression Classification Plot')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# from mpl_toolkits.mplot3d import Axes3D

# # Generate synthetic data
# X, y = make_classification(n_features=3, n_redundant=0, n_informative=3,
#                            n_clusters_per_class=1, random_state=42)

# # Fit logistic regression model
# model = LogisticRegression()
# model.fit(X, y)

# # Generate grid points
# x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# x3_min, x3_max = X[:, 2].min() - 1, X[:, 2].max() + 1
# xx1, xx2, xx3 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
#                             np.arange(x2_min, x2_max, 0.1),
#                             np.arange(x3_min, x3_max, 0.1))

# # Predict class probabilities for grid points
# X_grid = np.c_[xx1.ravel(), xx2.ravel(), xx3.ravel()]
# Z = model.predict(X_grid)
# Z = Z.reshape(xx1.shape)

# # Plot decision boundary and data points
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, edgecolors='k')
# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
# ax.set_zlabel('Feature 3')
# ax.set_title('Logistic Regression Classification Plot')
# plt.show()

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression

# # import some data to play with
# iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features.
# Y = iris.target

# Create an instance of Logistic Regression Classifier and fit the data.
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)

# # Visualize decision boundary and classes
# _, ax = plt.subplots(figsize=(8, 6))

# DecisionBoundaryDisplay.from_estimator(
#     logreg,
#     X_train,
#     cmap=plt.cm.Paired,
#     ax=ax,
#     response_method="predict",
#     plot_method="contourf",
#     eps=0.5
# )

# # Plot the training data points
# for i, class_label in enumerate(np.unique(y_train)):
#     ax.scatter(X_train[y_train == class_label, 0], X_train[y_train == class_label, 1], label=f'Class {class_label}', edgecolors='k')

# # Labeling
# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
# ax.set_title('Logistic Regression Decision Boundary and Classes')
# plt.legend()

# # Prediction
# predict_value = [27, 90000]  # Sample prediction value
# predict_value = sc.transform([predict_value])
# prediction_results = logreg.predict(predict_value)
# predicted_class = prediction_results[0]
# print(predicted_class)
# # Plot predicted point
# marker = 's' if predicted_class == 0 else 'o'  # Use square for class 0, circle for class 1
# color = 'red' if predicted_class == 0 else 'blue'  # Color for predicted point
# ax.scatter(predict_value[0][0], predict_value[0][1], c=color, marker=marker, s=100, label=f'Predicted Class {predicted_class}')

# plt.axis("tight")
# plt.legend()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
import seaborn as sns

# Generate synthetic data for demonstration
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Decision Boundaries (for 2D data)
if X_train.shape[1] == 2:
    h = .02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundaries')
    plt.show()
