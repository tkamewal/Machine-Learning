from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics  import accuracy_score, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import seaborn as sns 


data = pd.read_csv("C:\\Users\\TANMAY KAMEWAL\\OneDrive\\Desktop\\MachineLearning\\Social_Network_Ads.csv")
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
sc = StandardScaler()
x = sc.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
naive = GaussianNB()
naive.fit(x_train, y_train)
y_pred = naive.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy*100)


# Confusion Matrix
y_pred =naive.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
y_prob = naive.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Class Probability Distributions
plt.figure(figsize=(12, 6))
for i in range(2):  # Assuming 2 classes
    sns.kdeplot(y_prob[y_test == i], label=f'Class {i}', shade=True)
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Class Probability Distributions')
plt.legend()
plt.show()

# Feature Importance or Probability Visualization (assuming Gaussian Naive Bayes)
plt.figure(figsize=(12, 6))
for i in range(2):  # Assuming 2 classes
    sns.kdeplot(x_test[y_test == i].mean(axis=0), label=f'Class {i}', shade=True)
plt.xlabel('Feature Mean')
plt.ylabel('Density')
plt.title('Feature Means by Class')
plt.legend()
plt.show()
