from scipy import stats
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(211)
x = stats.loggamma.rvs(5, size=1000) + 5
prob = stats.probplot(x, dist="norm", plot=ax)
ax.set_xlabel("")
ax.set_title('probplot against normal distribution...')
ax2 = fig.add_subplot(212)
xt, _ = stats.boxcox(x)
prob = stats.probplot(xt, dist="norm", plot=ax2)
ax2.set_title('probplot after Box-Cox Transformation...')
plt.show()
