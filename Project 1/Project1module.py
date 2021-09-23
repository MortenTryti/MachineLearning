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


# making the OCS regression
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


def plotMSEcomplexity(x, y, z, n):
    MSElisttest = []
    MSElisttrain = []
    n = n + 1
    complexity = np.arange(n)
    print(f"The polynomial range is [{complexity[0]},{complexity[-1]}] ")

    # Running over the degrees of polynomails
    for degree in complexity:
        # Creating the designmatrix and splitting into train and test
        X = create_X(x, y, degree)
        X_train, X_test, z_train, z_test = train_test_split(X, z.reshape(-1, 1), test_size=0.2)
        MSEdeglisttest = []
        MSEdeglisttrain = []
        # Running over the bootstrap for the specific degree
        for i in range(22):
            # Using the bootstrap method to "create" more trainingdata
            bootX, bootz = resample(X_train, z_train.reshape(-1, 1))

            # Using the "new" data to calculate the coefficients
            bootbetaOLS = OLSmethod(bootX, bootz)

            # Making out model and adding it to a list
            ztilde = X_test @ bootbetaOLS
            zpred = X_train @ bootbetaOLS
            MSEdeglisttest.append(MSE(z_test, ztilde))
            MSEdeglisttrain.append(MSE(z_train, zpred))

        # Appending the mean to the MSE list when the loop has run for its specific degree
        MSElisttest.append(np.mean(np.array(MSEdeglisttest)))
        MSElisttrain.append(np.mean(np.array(MSEdeglisttrain)))

    plt.plot(complexity, MSElisttest, "r", label="test")
    plt.plot(complexity, MSElisttrain, "k", label="train")
    plt.xlabel("Polynomial degree")
    plt.grid()
    plt.ylabel("MSE")
    plt.title("Figure of the MSE as a function of the complexity of the model")
    plt.legend()
    plt.show()


def confidense(beta, X):
    conf = np.zeros(len(beta))
    betavar = np.var(beta)
    XtX = X.T @ X
    for i in range(len(conf)):
        conf[i] = betavar * XtX[i, i]
    return conf