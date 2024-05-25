import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv("haberman.csv")

sns.jointplot(x="OpYear", y="Age", data=data, kind='kde')
plt.show()