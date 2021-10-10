# MachineLearning - Fys-Stk3155/4155

### Groupmembers: Morten Tryti Berg, Henrik Haugerud Carlsen, Jonas Rønning


# README.md Project1
Project1.ipynb answers problems 1-5.
Task 6.ipynb answers problem 6.

All the figures in the pdf can be found in the figure directory
The .tif file is a map of a little piece of Norway outside of Stavanger.

The Report.pdf file is the full report for project 1.

### Prerequisites



### Functions
The functions of our script is as follows
#### FrankeFunction(x,y)
A function used to simulate a set of data we wish to fit
 
 #### R2(y_data, y_model)
Calculates the R2 value ofa fit and the data
 
 #### MSE(y,ytilde)
Calculates the MSE between a fit and the data

#### create_X(x, y, n )
Creates a n-th order polynomial with crossterms
  
#### OLSmethod(X,z)
Does an ordinary least squares fit between the data and the designmatrix, returns the optimal coefficients
  
#### ridgeregg(X,y, lmb = 0.0001)
Does a ridge regression between a desigmatrix and the data, returns the optimal cofficients
  
#### lassoregg(X,y,lmb = 0.0001)
Does a ridge regression between a desigmatrix and the data, returns the optimal cofficients

#### surfaceplot(x,y,z)
Plots a 3D surface of the Frankie function
  
#### printQ(xdata,xmodel)
 A function that prints the MSE and R2 in a neat way

#### BootstrapOLS(X_train,X_test,z_train,z_test,numberOfStraps)
Calculating the MSE, bias and variance for OLS using the bootstrap method, returns the average values of a number of bootstraps

#### BootstrapRidge(X_train,X_test,z_train,z_test,lam,numberOfStraps)
Calculating the MSE, bias and variance for Ridge regression using the bootstrap method, returns the average values of a number of bootstraps

#### BootstrapLasso(X_train,X_test,z_train,z_test,lam,numberOfStraps)
Calculating the MSE, bias and variance for Lasso regression using the bootstrap method, returns the average values of a number of bootstraps
  
  
#### plotMSEcomplexity(x,y,z,n):
A function for plotting the MSE for the OLS method with bootstrap for a given n complexity
  
#### confidense(y,X)
Finding the confidence intervall for beta
 
 #### biassVariance(y,y_pred)
Finding the bias and the variance

#### k_foldOLS
A function that returns the MSE for a k fold analysis of the OLS regression

#### k_foldRigd
A function that returns the MSE for a k fold analysis of the Ridge regression

#### k_foldLasso
A function that returns the MSE for a k fold analysis of the Lasso regression



### Usage

Run the programs by running the jupyter notebooks. Some notebooks may require the user to run the notebook locally in order to not run out of memory.
