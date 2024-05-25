import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Social_Network_Ads.csv")
# print(data.head())
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = data.iloc[:, [0, 1]].values
y = data.iloc[:, -1].values
# X = sc.fit_transform(X)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC

classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# from sklearn.metrics import accuracy_score

# print(accuracy_score(y_test, y_pred))
# print(classifier.predict(sc.transform([[27, 90000]])))
# from matplotlib.colors import ListedColormap

# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
#                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha=0.75, cmap=ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c=ListedColormap(('red', 'green'))(i), label=j)
#     plt.title('SVM (Training set)')
#     plt.xlabel('Age')
#     plt.ylabel('Estimated Salary')
#     plt.legend()
#     plt.show()

# from matplotlib.colors import ListedColormap

# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
#                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha=0.75, cmap=ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c=ListedColormap(('red', 'green'))(i), label=j)
#     plt.title('SVM (Test set)')
#     plt.xlabel('Age')
#     plt.ylabel('Estimated Salary')
#     plt.legend()
#     plt.show()


# figure number
# fignum = 1
# plt.figure(figsize=(15,10))
# # fit the model
# for name, penalty in (("unreg", 1), ("reg", 0.05)):
#     clf = SVC(kernel="linear", C=penalty)
#     clf.fit(X_train, y_train)

#     # get the separating hyperplane
#     w = clf.coef_[0]
#     a = -w[0] / w[1]
#     xx = np.linspace(-5, 5)
#     yy = a * xx - (clf.intercept_[0]) / w[1]

#     # plot the parallels to the separating hyperplane that pass through the
#     # support vectors (margin away from hyperplane in direction
#     # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
#     # 2-d.
#     margin = 1 / np.sqrt(np.sum(clf.coef_**2))
#     yy_down = yy - np.sqrt(1 + a**2) * margin
#     yy_up = yy + np.sqrt(1 + a**2) * margin

#     # plot the line, the points, and the nearest vectors to the plane
#     plt.figure(fignum, figsize=(4, 3))
#     plt.clf()
#     plt.plot(xx, yy, "k-")
#     plt.plot(xx, yy_down, "k--")
#     plt.plot(xx, yy_up, "k--")

#     plt.scatter(
#         clf.support_vectors_[:, 0],
#         clf.support_vectors_[:, 1],
#         s=80,
#         facecolors="none",
#         zorder=10,
#         edgecolors="k",
#         cmap=plt.get_cmap("RdBu"),
#     )
#     plt.scatter(
#         X_train[:, 0], X_train[:, 1], c=y_train, zorder=10, cmap=plt.get_cmap("RdBu"), edgecolors="k"
#     )

#     plt.axis("tight")
#     # x_min = -4.8
#     # x_max = 4.2
#     # y_min = -6
#     # y_max = 6

#     YY, XX = np.meshgrid(yy, xx)
#     xy = np.vstack([XX.ravel(), YY.ravel()]).T
#     Z = clf.decision_function(xy).reshape(XX.shape)

#     # Put the result into a contour plot
#     plt.contourf(XX, YY, Z, cmap=plt.get_cmap("RdBu"), alpha=0.5, linestyles=["-"])

#     # plt.xlim(x_min, x_max)
#     # plt.ylim(y_min, y_max)

#     plt.xticks(())
#     plt.yticks(())
#     fignum = fignum + 1

# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# from sklearn import svm

# # we create 40 separable points
# np.random.seed(0)
# X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
# Y = [0] * 20 + [1] * 20

# # figure number
# fignum = 1

# # fit the model
# for name, penalty in (("unreg", 1), ("reg", 0.05)):
#     clf = svm.SVC(kernel="linear", C=penalty)
#     clf.fit(X, Y)

#     # get the separating hyperplane
#     w = clf.coef_[0]
#     a = -w[0] / w[1]
#     xx = np.linspace(-5, 5)
#     yy = a * xx - (clf.intercept_[0]) / w[1]

#     # plot the parallels to the separating hyperplane that pass through the
#     # support vectors (margin away from hyperplane in direction
#     # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
#     # 2-d.
#     margin = 1 / np.sqrt(np.sum(clf.coef_**2))
#     yy_down = yy - np.sqrt(1 + a**2) * margin
#     yy_up = yy + np.sqrt(1 + a**2) * margin

#     # plot the line, the points, and the nearest vectors to the plane
#     plt.figure(fignum, figsize=(4, 3))
#     plt.clf()
#     plt.plot(xx, yy, "k-")
#     plt.plot(xx, yy_down, "k--")
#     plt.plot(xx, yy_up, "k--")

#     plt.scatter(
#         clf.support_vectors_[:, 0],
#         clf.support_vectors_[:, 1],
#         s=80,
#         facecolors="none",
#         zorder=10,
#         edgecolors="k",
#         cmap=plt.get_cmap("RdBu"),
#     )
#     plt.scatter(
#         X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.get_cmap("RdBu"), edgecolors="k"
#     )

#     plt.axis("tight")
#     x_min = -4.8
#     x_max = 4.2
#     y_min = -6
#     y_max = 6

#     YY, XX = np.meshgrid(yy, xx)
#     xy = np.vstack([XX.ravel(), YY.ravel()]).T
#     Z = clf.decision_function(xy).reshape(XX.shape)

#     # Put the result into a contour plot
#     plt.contourf(XX, YY, Z, cmap=plt.get_cmap("RdBu"), alpha=0.5, linestyles=["-"])

#     plt.xlim(x_min, x_max)
#     plt.ylim(y_min, y_max)

#     plt.xticks(())
#     plt.yticks(())
#     fignum = fignum + 1

# plt.show()

import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay

# we create 40 separable points
# X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# fit the model, don't regularize for illustration purposes
# clf = svm.SVC(kernel="linear", C=1000)
# clf.fit(X, y)

# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, cmap=plt.cm.Paired)

# # plot the decision function
# ax = plt.gca()
# DecisionBoundaryDisplay.from_estimator(
#     classifier,
#     X_train,
#     plot_method="contour",
#     colors="k",
#     levels=[-1, 0, 1],
#     alpha=0.5,
#     linestyles=["--", "-", "--"],
#     ax=ax,
# )
# # plot support vectors
# ax.scatter(
#     classifier.support_vectors_[:, 0],
#     classifier.support_vectors_[:, 1],
#     s=100,
#     linewidth=1,
#     facecolors="none",
#     edgecolors="k",
# )
# plt.show()

# Visualize decision boundary and classes
# _, ax = plt.subplots(figsize=(8, 6))

# DecisionBoundaryDisplay.from_estimator(
#         classifier,
#         X_train,
#         cmap=plt.cm.Paired,
#         ax=ax,
#         response_method="predict",
#         plot_method="contourf",
#         eps=0.5
#     )

#     # Plot the training data points
# for i, class_label in enumerate(np.unique(y_train)):
#     ax.scatter(X_train[y_train == class_label, 0], X_train[y_train == class_label, 1], label=f'Class {class_label}',
#                    edgecolors='k')

#     # # Labeling
#     # ax.set_xlabel(input_column[0])
#     # ax.set_ylabel(input_column[1])
# ax.set_title('Logistic Regression Decision Boundary and Classes')
# plt.legend()

#     # pre = sc.transform([[32, 12221]])
#     # # Plot predicted point
#     # marker = 's' if prediction_results[0] == 0 else '*'  # Use square for class 0, circle for class 1
#     # color = 'red' if prediction_results[0] == 0 else 'blue'  # Color for predicted point
#     # ax.scatter(pre[0][0], pre[0][1], c=color, marker=marker, s=100,
#     #            label=f'Predicted Class {prediction_results[0]}')

# plt.axis("tight")
# plt.legend()

# plt.tight_layout()
# plt.show()

# def plot_svc_decision_function(model, ax=None, plot_support=True):
#     """Plot the decision function for a 2D SVC"""
#     if ax is None:
#         ax = plt.gca()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
    
#     # create grid to evaluate model
#     # x = np.linspace(xlim[0], xlim[1], 30)
#     # y = np.linspace(ylim[0], ylim[1], 30)
#     Y, X = np.meshgrid(y_train, X_train)
#     xy = np.vstack([X.ravel(), Y.ravel()]).T
#     P = model.decision_function(xy).reshape(X.shape)
    
#     # plot decision boundary and margins
#     ax.contour(X, Y, P, colors='k',
#                levels=[-1, 0, 1], alpha=0.5,
#                linestyles=['--', '-', '--'])
    
#     # plot support vectors
#     if plot_support:
#         ax.scatter(model.support_vectors_[:, 0],
#                    model.support_vectors_[:, 1],
#                    s=300, linewidth=1, facecolors='none');
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
#     plt.show()

# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter')
# plot_svc_decision_function(classifier);    

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import svm, datasets

# # Load example dataset
# iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features.
# y = iris.target

# # Create SVC model
# svc = svm.SVC(kernel='linear', C=1.0)
# svc.fit(X, y)

# # Plot the decision boundary with margins
# plt.figure(figsize=(8, 6))

# # Plot data points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, label='Data Points')

# # Plot decision boundaries
# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()

# # Create grid to evaluate model
# xx = np.linspace(xlim[0], xlim[1], 30)
# yy = np.linspace(ylim[0], ylim[1], 30)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = svc.decision_function(xy)

# # Plot decision boundaries and margins
# ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
# ax.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')
# plt.legend()
# plt.show()

# # Standard scientific Python imports
# from time import time

# import matplotlib.pyplot as plt
# import numpy as np

# # Import datasets, classifiers and performance metrics
# from sklearn import datasets, pipeline, svm
# from sklearn.decomposition import PCA
# from sklearn.kernel_approximation import Nystroem, RBFSampler

# # The digits dataset
# digits = datasets.load_digits(n_class=9)

# n_samples = len(digits.data)
# data = digits.data / 16.0
# data -= data.mean(axis=0)

# # We learn the digits on the first half of the digits
# data_train, targets_train = (data[: n_samples // 2], digits.target[: n_samples // 2])


# # Now predict the value of the digit on the second half:
# data_test, targets_test = (data[n_samples // 2 :], digits.target[n_samples // 2 :])
# # data_test = scaler.transform(data_test)

# # Create a classifier: a support vector classifier
# kernel_svm = svm.SVC(gamma=0.2)
# linear_svm = svm.LinearSVC(dual="auto", random_state=42)

# # create pipeline from kernel approximation
# # and linear svm
# feature_map_fourier = RBFSampler(gamma=0.2, random_state=1)
# feature_map_nystroem = Nystroem(gamma=0.2, random_state=1)
# fourier_approx_svm = pipeline.Pipeline(
#     [
#         ("feature_map", feature_map_fourier),
#         ("svm", svm.LinearSVC(dual="auto", random_state=42)),
#     ]
# )

# nystroem_approx_svm = pipeline.Pipeline(
#     [
#         ("feature_map", feature_map_nystroem),
#         ("svm", svm.LinearSVC(dual="auto", random_state=42)),
#     ]
# )

# # fit and predict using linear and kernel svm:

# kernel_svm_time = time()
# kernel_svm.fit(data_train, targets_train)
# kernel_svm_score = kernel_svm.score(data_test, targets_test)
# kernel_svm_time = time() - kernel_svm_time

# linear_svm_time = time()
# linear_svm.fit(data_train, targets_train)
# linear_svm_score = linear_svm.score(data_test, targets_test)
# linear_svm_time = time() - linear_svm_time

# sample_sizes = 30 * np.arange(1, 10)
# fourier_scores = []
# nystroem_scores = []
# fourier_times = []
# nystroem_times = []

# for D in sample_sizes:
#     fourier_approx_svm.set_params(feature_map__n_components=D)
#     nystroem_approx_svm.set_params(feature_map__n_components=D)
#     start = time()
#     nystroem_approx_svm.fit(data_train, targets_train)
#     nystroem_times.append(time() - start)

#     start = time()
#     fourier_approx_svm.fit(data_train, targets_train)
#     fourier_times.append(time() - start)

#     fourier_score = fourier_approx_svm.score(data_test, targets_test)
#     nystroem_score = nystroem_approx_svm.score(data_test, targets_test)
#     nystroem_scores.append(nystroem_score)
#     fourier_scores.append(fourier_score)

# # plot the results:
# plt.figure(figsize=(16, 4))
# accuracy = plt.subplot(121)
# # second y axis for timings
# timescale = plt.subplot(122)

# accuracy.plot(sample_sizes, nystroem_scores, label="Nystroem approx. kernel")
# timescale.plot(sample_sizes, nystroem_times, "--", label="Nystroem approx. kernel")

# accuracy.plot(sample_sizes, fourier_scores, label="Fourier approx. kernel")
# timescale.plot(sample_sizes, fourier_times, "--", label="Fourier approx. kernel")

# # horizontal lines for exact rbf and linear kernels:
# accuracy.plot(
#     [sample_sizes[0], sample_sizes[-1]],
#     [linear_svm_score, linear_svm_score],
#     label="linear svm",
# )
# timescale.plot(
#     [sample_sizes[0], sample_sizes[-1]],
#     [linear_svm_time, linear_svm_time],
#     "--",
#     label="linear svm",
# )

# accuracy.plot(
#     [sample_sizes[0], sample_sizes[-1]],
#     [kernel_svm_score, kernel_svm_score],
#     label="rbf svm",
# )
# timescale.plot(
#     [sample_sizes[0], sample_sizes[-1]],
#     [kernel_svm_time, kernel_svm_time],
#     "--",
#     label="rbf svm",
# )

# # vertical line for dataset dimensionality = 64
# accuracy.plot([64, 64], [0.7, 1], label="n_features")

# # legends and labels
# accuracy.set_title("Classification accuracy")
# timescale.set_title("Training times")
# accuracy.set_xlim(sample_sizes[0], sample_sizes[-1])
# accuracy.set_xticks(())
# accuracy.set_ylim(np.min(fourier_scores), 1)
# timescale.set_xlabel("Sampling steps = transformed feature dimension")
# accuracy.set_ylabel("Classification accuracy")
# timescale.set_ylabel("Training time in seconds")
# accuracy.legend(loc="best")
# timescale.legend(loc="best")
# plt.tight_layout()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import datasets
# from sklearn.svm import SVC

# # Load dataset
# iris = datasets.load_iris()
# X = iris.data[:, :2]  # Taking only the first two features for visualization purposes
# y = iris.target

# # Train SVM with a linear kernel
# svm_model = SVC(kernel='linear', C=1)
# svm_model.fit(X, y)

# # Function to plot decision boundaries, support vectors, and margin
# def plot_svm_decision_boundary(X, y, model):
#     plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

#     # plot the decision function
#     ax = plt.gca()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()

#     # create grid to evaluate model
#     xx = np.linspace(xlim[0], xlim[1], 30)
#     yy = np.linspace(ylim[0], ylim[1], 30)
#     YY, XX = np.meshgrid(yy, xx)
#     xy = np.vstack([XX.ravel(), YY.ravel()]).T
#     Z = model.decision_function(xy).reshape(XX.shape)

#     # plot decision boundary and margins
#     ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

#     # plot support vectors
#     ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

#     plt.xlabel('Sepal length')
#     plt.ylabel('Sepal width')
#     plt.title('SVM Decision Boundary')

# # Plot SVM decision boundary with support vectors
# plt.figure(figsize=(10, 6))
# plot_svm_decision_boundary(X, y, svm_model)
# plt.show()

# print(__doc__)

# import numpy as np
# from scipy import interp
# import matplotlib.pyplot as plt
# from itertools import cycle

# from sklearn import svm, datasets
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import StratifiedKFold

# # #############################################################################
# # Data IO and generation

# # Import some data to play with
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# X, y = X[y != 2], y[y != 2]
# n_samples, n_features = X.shape

# # Add noisy features
# random_state = np.random.RandomState(0)
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# # #############################################################################
# # Classification and ROC analysis

# # Run classifier with cross-validation and plot ROC curves
# cv = StratifiedKFold(n_splits=6)
# classifier = svm.SVC(kernel='linear', probability=True,
#                      random_state=random_state)

# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)

# i = 0
# for train, test in cv.split(X, y):
#     probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
#     # Compute ROC curve and area the curve
#     fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
#     tprs.append(interp(mean_fpr, fpr, tpr))
#     tprs[-1][0] = 0.0
#     roc_auc = auc(fpr, tpr)
#     aucs.append(roc_auc)
#     plt.plot(fpr, tpr, lw=1, alpha=0.3,
#              label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

#     i += 1
# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#          label='Chance', alpha=.8)

# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# plt.plot(mean_fpr, mean_tpr, color='b',
#          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#          lw=2, alpha=.8)

# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                  label=r'$\pm$ 1 std. dev.')

# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


# we create 40 separable points
# X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X_train, y_train)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()
