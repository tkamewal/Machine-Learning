import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv("plant.csv")
iris.plot(kind='scatter', x='SepalLength', y='SepalWidth')
# plt.show()

sns.set_style('whitegrid')
sns.FacetGrid(iris, hue='Species', height=4).map(plt.scatter, 'SepalLength', 'SepalWidth').add_legend()
# plt.show()


iris.plot(kind='scatter', x='PetalLength', y='PetalWidth')
# plt.show()

sns.set_style('whitegrid')
sns.FacetGrid(iris, hue='Species', height=4).map(plt.scatter, 'PetalLength', 'PetalWidth').add_legend()
# plt.show()

# Pair Plot
sns.set_style('whitegrid')
sns.pairplot(iris, hue='Species',height=3)
plt.show()
