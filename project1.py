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
from statistics import NormalDist

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures



# FrankeFunction
def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)

    return term1 + term2 + term3 + term4


# Defining the R2 function
def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


# Defining the Mean square error
def MSE(y, ytilde):
    n = len(y)
    return 1 / n * np.sum(np.abs(y - ytilde) ** 2)


# Creating the design matrix
def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x ** (i - k)) * (y ** k)

    return X


# making the OLS regression
def OLSmethod(X, z):
    return np.linalg.pinv(X.T @ X) @ X.T @ z


# Ridgeregression
def ridgeregg(X, y, lmb=0.0001):
    XtX = X.T @ X
    return np.linalg.pinv(XtX + lmb * np.identity(len(y))) @ X.T @ y


def surfaceplot(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def printQ(xdata, xmodel):
    print("--------------------------------------------------------")
    print(f"MSE = |{MSE(xdata, xmodel)}|, R2 = |{r2_score(xdata, xmodel)}|")
    print("--------------------------------------------------------\n")


def bootstrap(X):
    return resample(X)


npoints = 100


# Make data.
x = np.sort(np.random.uniform(0, 1, npoints))
y = np.sort(np.random.uniform(0, 1, npoints))


x, y = np.meshgrid(x,y)


z = FrankeFunction(x, y)
z = z #+ 0.1 * np.random.randn(z.shape[0])

#surfaceplot(x,y,z)

n = 6

x = np.sort(np.random.uniform(0.0, 1.0, npoints))
y = np.sort(np.random.uniform(0.0, 1.0, npoints))

x, y = np.meshgrid(x, y)

X = create_X(x, y, n)

print(X.shape, z.shape)

X_train, X_test, z_train, z_test = train_test_split(X, z.reshape(-1, 1), test_size=0.2)

betaOLS = OLSmethod(X_train, z_train)

print(np.var(betaOLS))
print(np.std(betaOLS))
print(betaOLS)

def conf_int(data,z):
    d = np.linalg.pinv(X.T @ X)
    beta_i_var = np.zeros(len(betaOLS))
    sigma = np.var(FrankeFunction(x,y))
    for i in range(len(betaOLS)):
        beta_i_var[i] = d[i,i] ###* sigma
    print(d[i,i])
    return beta_i_var
confidence = conf_int(X, 1.96)
print(confidence)





zpred = X_train @ betaOLS
print("For the training data we get a MSE of")
printQ(z_train, zpred)

print("For the test data we get a MSE of")
ztilde = X_test @ betaOLS
printQ(z_test, ztilde)

MSElist = []
R2list = []
for i in range(28):
    bootX, bootz = resample(X, z.reshape(-1, 1))

    X_train, X_test, z_train, z_test = train_test_split(bootX, bootz, test_size=0.2)

    bootbetaOLS = OLSmethod(X_train, z_train)

    ypred = X_train @ bootbetaOLS
    MSElist.append(MSE(z_train, ypred))
    R2list.append(R2(z_train, ypred))

print(np.mean(MSElist))
print(np.mean(R2list))



scaler = StandardScaler()

X_scale = scaler.fit_transform(X)
z_scaled = scaler.fit_transform(z)





X_strain, X_stest, z_strain, z_stest = train_test_split(X_scale, z_scaled.reshape(-1,1) , test_size = 0.2 )

betascaledOCS = np.linalg.pinv(X_strain.T@X_strain)@ X_strain.T @ z_strain
print(f"The variance of the scaled coefficients is {np.var(betascaledOCS)}\n")

zpreds = X_strain@betascaledOCS



print(f"Under we print the different values for MSE and R2")



print("For the scaled training data we get a MSE and R2 of:")
printQ(z_strain,zpreds)


print("For the scaled test data we get a MSE and R2 of: ")
ztilde = X_stest @ betascaledOCS

printQ(z_stest,ztilde)



print(np.shape(X))
print(np.shape(z))

"""Forsøker å splitte dataen z=Frankefunction(x,y) til 5 folds og så finne MSE ved å bruke OLSmethod
av de dedikerte train og test settene.

Skjønner ikke hvorfor Morten, i sitt eksempel lager random data x_r som ha deler opp sammen med dataen shrink.

"""

x_r = np.random.randn(100)
def k_fold(Data, k, Func):

    "Splitting the data"
    k_split = KFold(n_splits = k)

    "CV to calculate MSE"
    k_scores= np.zeros(k)

    i = 0
    for k_train_index, k_test_index in k_split.split(Data):
        k_xtrain = Data[k_train_index]
        k_ytrain = x_r[k_train_index]

        k_xtest = Data[k_test_index]
        k_ytest = x_r[k_test_index]

        #k_Xtrain = poly.fit_transform(k_xtrain[:, np.newaxis])
        "Finding betaOLS for each k"
        k_OLS = OLSmethod(k_xtrain, k_ytrain)

        #k_Xtest = poly.fit_transform(k_xtest[:, np.newaxis])
        model_predict = k_xtest @ k_OLS

        k_scores[i] = MSE(k_ytest, model_predict)

    i += 1
    MSE_kfold = np.mean(k_scores)
    print('MSE for k-fold OLS')
    print(MSE_kfold)

k_fold(z, 5, x_r)
