import numpy as np
import pylab
import scipy.stats as stats

measurment = np.random.normal(loc=20, scale=5, size=10000)
stats.probplot(measurment, dist="norm", plot=pylab)
pylab.show()