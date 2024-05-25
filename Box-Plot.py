import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
iris = pd.read_csv("plant.csv")
sns.boxplot(x="Species", y="PetalLength", data=iris)
plt.show()

