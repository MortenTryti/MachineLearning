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


npoints = 100


# Make data.
x = np.sort(np.random.uniform(0, 1, npoints))
y = np.sort(np.random.uniform(0, 1, npoints))


x, y = np.meshgrid(x,y)


z = p1.FrankeFunction(x, y)
z = z + 0.1 * np.random.randn(z.shape[0])

p1.surfaceplot(x,y,z)

n = 5

x = np.sort(np.random.uniform(0.0, 1.0, npoints))
y = np.sort(np.random.uniform(0.0, 1.0, npoints))

x, y = np.meshgrid(x, y)

X = p1.create_X(x, y, n)

X_train, X_test, z_train, z_test = train_test_split(X, z.reshape(-1, 1), test_size=0.2)

betaOLS = p1.OLSmethod(X_train, z_train)

print(f"The variance is {np.var(betaOLS)}")
print(np.std(betaOLS))

zpred = X_train @ betaOLS
print("For the training data we get a MSE of")
p1.printQ(z_train, zpred)

print("For the test data we get a MSE of")
ztilde = X_test @ betaOLS
p1.printQ(z_test, ztilde)

MSElist = []
R2list = []
for i in range(28):
    bootX, bootz = resample(X_train, z_train.reshape(-1, 1))

    bootbetaOLS = p1.OLSmethod(bootX, bootz)

    ypred = bootX @ bootbetaOLS
    MSElist.append(p1.MSE(z_train, ypred))
    R2list.append(p1.R2(bootz, ypred))


scaler = StandardScaler()

X_scale = scaler.fit_transform(X)
z_scaled = scaler.fit_transform(z)




X_strain, X_stest, z_strain, z_stest = train_test_split(X_scale, z_scaled.reshape(-1,1) , test_size = 0.2 )

betascaledOCS = np.linalg.pinv(X_strain.T@X_strain)@ X_strain.T @ z_strain
print(f"The variance of the scaled coefficients is {np.var(betascaledOCS)}\n")

zpreds = X_strain@betascaledOCS



print(f"Under we print the different values for MSE and R2")



print("For the scaled training data we get a MSE and R2 of:")
p1.printQ(z_strain,zpreds)


print("For the scaled test data we get a MSE and R2 of: ")
ztilde = X_stest @ betascaledOCS

p1.printQ(z_stest,ztilde)


print(np.mean(MSElist))
print(np.mean(R2list))