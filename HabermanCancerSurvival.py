# survival 1-will live 5 years or longer  2-will die within 5 years
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('haberman.csv')
print(data.head())

print(data.shape)
data["Survival"] = data["Survival"].map({1: "Survived", 2: "Died"})
print(data.head())
print(data.describe())
# sns.FacetGrid(data, hue="Survival", height=5).map(sns.distplot, 'Age').add_legend()
# sns.FacetGrid(data, hue='Survival', height=4).map(plt.scatter, 'Age','NodesDetect').add_legend()
# sns.FacetGrid(data, hue="Survival", height=5).map(sns.distplot, 'OpYear').add_legend()
# sns.FacetGrid(data, hue="Survival", height=5).map(sns.distplot, 'NodesDetect').add_legend()
# sns.FacetGrid(data, hue="Survival", height=5).map(sns.distplot, 'Survival').add_legend()
# sns.pairplot(data, hue="Survival")
# survival_Yes = data[data.Survival == "Survived"]
# survival_No = data[data.Survival == "Died"]

# counts1, bins1 = np.histogram(survival_Yes["Age"], bins=10, density=True)
# counts2, bins2 = np.histogram(survival_No["Age"], bins=10, density=True)
# pdf1 = counts1 / np.sum(counts1)
# pdf2 = counts2 / np.sum(counts2)
# print(pdf1)
# print(bins1)
# print(pdf2)
# print(bins2)
# cdf1 = np.cumsum(pdf1)
# cdf2 = np.cumsum(pdf2)
# plt.plot(bins1[1:], pdf1)
# plt.plot(bins1[1:], cdf1)
# plt.plot(bins2[1:], pdf2)
# plt.plot(bins2[1:], cdf2)
# plt.xlabel("Age")
# plt.ylabel("Probability")
# plt.legend(["Survived", "Died"])
# plt.title("Haberman Cancer Survival")
sns.boxplot(x="Survival", y="Age", data=data)
sns.violinplot(x="Survival", y="Age", data=data)
plt.show()



