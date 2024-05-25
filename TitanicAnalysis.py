import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Titanic_Data.csv")
print(data.head())
print(data.columns)
# data["Survived_yes"] = data["Survived"]==1
# data["Survived_no"] = data["Survived"]==0
sns.FacetGrid(data, hue="Survived").map(plt.scatter, "Age", "Sex").add_legend()
plt.show()
