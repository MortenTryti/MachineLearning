# MachineLearning - Fys-Stk3155/4155

### Groupmembers: Morten Tryti Berg, Henrik Haugerud Carlsen, Jonas Rønning


# README.md Project3
For our project number 3 we choose to do the suggested task, which was to solve the diffusion equation analytically, numerically and with a neural network as well as 
solving an eigenvalue problem. 

P3_taskb.ipynb contains all the code for solving problem b).

Our_network.ipynb contains all the code for solving problems c and d).

The Report.pdf file is the full report for project 3.

All the figures in the pdf can be found in the figure directory.





### Functions used in the tasks:

### Sigmoid(y)
Calculates the function values of the sigmoid activation function.


### def Identity(y):
Applies the identity function to input y, so nothing happens.


### RELU(y)
Calculates the function values of the RELU activation function.


### lexyRelu(y)
Calculates the function values of the Leaky-RELU activation function.


### Set_weights_and_bias(n_in,n_hidden,n_out)
Takes number of input nodes, number of hidden nodes and number of outputnodes.
Returns Weights for hidden layer, weights for the output layer, the output for the hidden layer(s) and the output for the output layer.



### feed_forward_train(X,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
(From lecture notes)

(Used in order to use our own backpropagation from project 2)
Takes as input the designmatrix, X, the weights for the output layer, the output for the hidden layer(s) and the output for the output layer, an activation function and an output function. Returns the output of the activation function in the hidden layer(s), the output of the output function on the output layer, the output from the hidden layer(s) and the output from the output layer.


### feed_forward(X,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
Does the same as feed_forward_train(), but only returns the output of the output function on the output layer as a scalar.

### feed_forward_eigen(X,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
Does the same as feed_forward() but returns a_o as a reshaped vector.


### u(x)
Returns the function value of sin(pi*x).


### g_trial(X,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
Uses feed_forward_eigen() and X to return a guess of trial function.



### cost_function(x,t,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
#(From lecture notes)
Calculates the cost function

### def g_trial_eigen(t,W_hidden, W_out, b_hidden, b_out,activation_function, output_function,x_0)
Returns the trial function for the eigenvalue problem.

### def eigen_func(g_t,A)
Calculates the right hand side of equation 20 in the report.

### def cost_function_eigen(t,W_hidden, W_out, b_hidden, b_out,activation_function, output_function,A,x_0):
Calculates the cost function for the eigenvalue problem.

### def solution(x,t,N_hidden,eta, activation_function ,output_function, epoch,Minibach)
Creates a network and returns the weights of the hidden layer, the weights for the output layer, the output from the hidden layer and the output from the output layer.

### def solution_eigen(t,N_hidden,eta, activation_function ,output_function, epoch,Minibach,A,x_0):
Creates a network and returns, for the eigenvalue problem, the weights of the hidden layer, the weights for the output layer, the output from the hidden layer and the output from the output layer.
   

### def Solution2(x,t,N_hidden,eta, epoch,Minibach, W_hidden, W_out, b_hidden, b_out):
Returns, for an already created network, the weights of the hidden layer, the weights for the output layer, the output from the hidden layer and the output from the output layer.
 
