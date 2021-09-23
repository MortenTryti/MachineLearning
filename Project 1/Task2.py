import Project1module as p1
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.utils import resample

npoints = 18


# Make data.
x = np.sort(np.random.uniform(0, 1, npoints))
y = np.sort(np.random.uniform(0, 1, npoints))


x, y = np.meshgrid(x,y)


z = p1.FrankeFunction(x, y)
z = z + 0.1 * np.random.randn(z.shape[0])


n = 12

p1.plotMSEcomplexity(x,y,z,n)