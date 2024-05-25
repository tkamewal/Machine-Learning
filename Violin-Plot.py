import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = pd.read_csv('plant.csv')
sns.set_style('whitegrid')
sns.violinplot(x='Species', y='PetalLength', data=iris)
plt.show()