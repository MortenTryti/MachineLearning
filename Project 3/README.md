# MachineLearning - Fys-Stk3155/4155

### Groupmembers: Morten Tryti Berg, Henrik Haugerud Carlsen, Jonas Rønning


# README.md Project3
P3_taskb.ipynb answers problem b).
Our_network.ipynb answers problem c and d).
The Report.pdf file is the full report for project 3.
All the figures in the pdf can be found in the figure directory.



### Prerequisites


### Functions:

### Sigmoid(y)
Calculates the function values of the sigmoid activation function.

### RELU(y)
Calculates the function values of the RELU activation function.


### lexyRelu(y)
Calculates the function values of the Leaky-RELU activation function.


### Set_weights_and_bias(n_in,n_hidden,n_out)
Takes number of input nodes, number of hidden nodes and number of outputnodes.
Returns Weights for hidden layer, weights for the output layer, the output for the hidden layer(s) and the output for the output layer.



### feed_forward_train(X,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
#(From lecture notes)
#(Used in order to use our own backpropagation form project 2)
Takes as input the designmatrix, X, the weights for the output layer, the output for the hidden layer(s) and the output for the output layer, an activation function and an output function. Returns the output of the activation function in the hidden layer(s), the output of the output function on the output layer, the output from the hidden layer(s) and the output from the output layer.


### feed_forward(X,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
Does the same as feed_forward_train(), but only returns the output from the output of the output function on the output layer as a scalar.


### feed_forward_eigen(X,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
Does the same as feed_forward() but returns a_o as a reshaped vector.


### u(x)
Returns the function value of sin(pi*x).

### g_trial(X,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
Uses feed_forward_eigen() and X to return a guess of trial function.
    


### cost_function(x,t,W_hidden, W_out, b_hidden, b_out,activation_function, output_function):
```
#(From lecture notes)
Calculates the cost function
    cost_sum = 0
    
    g_t_jac_fun = jacobian(g_trial)
    g_t_hessian_fun = hessian(g_trial)
    
    for x_ in x:
        for t_ in t:
            X = np.array([x_,t_])
            g_t = g_trial(X,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
            g_t_jac = g_t_jac_fun(X,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
            g_t_hessian =g_t_hessian_fun(X,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
           # print(g_t)
            g_t_dt = g_t_jac[1]
            g_t_d2x = g_t_hessian[0][0]
            
            error = g_t_dt -g_t_d2x
            cost_sum += error**2
    return cost_sum/(np.size(x)*np.size(t))
```
### def g_trial_eigen(t,W_hidden, W_out, b_hidden, b_out,activation_function, output_function,x_0):
```
    return x_0 + t*(feed_forward_eigen(t,W_hidden, W_out, b_hidden, b_out,activation_function, output_function))
```
### def eigen_func(g_t,A):
```
    return (g_t.T@g_t)[0] *A@ g_t - (g_t.T@ A @g_t)[0]*g_t ## Python cant recognice scalars
```
### def cost_function_eigen(t,W_hidden, W_out, b_hidden, b_out,activation_function, output_function,A,x_0):
```
    g_t_dt_func = elementwise_grad(g_trial_eigen,0)
    square_error =0
    
    for t_ in t:
        g_t = g_trial_eigen(t_,W_hidden, W_out, b_hidden, b_out,activation_function, output_function,x_0)
        g_t_dt = g_t_dt_func(t_,W_hidden, W_out, b_hidden, b_out,activation_function, output_function,x_0)
     
        square_error += (g_t_dt - eigen_func(g_t,A))**2
    
    return square_error/np.size(t)
```

### from notes, altered to fit our setup
### def solution(x,t,N_hidden,eta, activation_function ,output_function, epoch,Minibach):
```
    W_hidden, W_out, b_hidden, b_out = Set_weights_and_bias(2,N_hidden,1)
    
    cost_func_wh_grad = elementwise_grad(cost_function,2)
    cost_func_bh_grad = elementwise_grad(cost_function,4)
    cost_func_wo_grad = elementwise_grad(cost_function,3)
    cost_func_bo_grad = elementwise_grad(cost_function,5)
    MiniBachSize =int(np.size(x)/Minibach)
    for e in range(epoch):
        for j in range(Minibach):
            miniBach = np.random.randint(Minibach)
            miniBachMin, miniBachMax = MiniBachSize * miniBach,(MiniBachSize) * (miniBach+1)
            x_bach,t_bach = x[miniBachMin:miniBachMax],t[miniBachMin:miniBachMax]
            W_hidden -= eta*cost_func_wh_grad(x_bach,t_bach,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
            W_out -= eta*cost_func_wo_grad(x_bach,t_bach,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
            b_hidden -= eta*cost_func_bh_grad(x_bach,t_bach,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
            b_out -= eta*cost_func_bo_grad(x_bach,t_bach,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
      
   
    return W_hidden, W_out, b_hidden, b_out
```
### def solution_eigen(t,N_hidden,eta, activation_function ,output_function, epoch,Minibach,A,x_0):
```
    W_hidden, W_out, b_hidden, b_out = Set_weights_and_bias(1,N_hidden,6)
 
    cost_function_eigen(t,W_hidden, W_out, b_hidden, b_out,activation_function, output_function,A,x_0)
    
    cost_func_wh_grad = elementwise_grad(cost_function_eigen,1)
    cost_func_bh_grad = elementwise_grad(cost_function_eigen,3)
    cost_func_wo_grad = elementwise_grad(cost_function_eigen,2)
    cost_func_bo_grad = elementwise_grad(cost_function_eigen,4)
    MiniBachSize =int(np.size(t)/Minibach)
    for e in range(epoch):
        for j in range(Minibach):
        #    print(j)
            miniBach = np.random.randint(Minibach)
            miniBachMin, miniBachMax = MiniBachSize * miniBach,(MiniBachSize) * (miniBach+1)
            t_bach = t[miniBachMin:miniBachMax]
            W_hidden -= eta*cost_func_wh_grad(t_bach,W_hidden, W_out, b_hidden, b_out,activation_function, output_function,A,x_0)
            W_out -= eta*cost_func_wo_grad(t_bach,W_hidden, W_out, b_hidden, b_out,activation_function, output_function,A,x_0)
            b_hidden -= eta*cost_func_bh_grad(t_bach,W_hidden, W_out, b_hidden, b_out,activation_function, output_function,A,x_0)
            b_out -= eta*cost_func_bo_grad(t_bach,W_hidden, W_out, b_hidden, b_out,activation_function, output_function,A,x_0)
      #  print("finished iteration number: ", e)
   
   
    return W_hidden, W_out, b_hidden, b_out
```

#### This one is nice when we want to study how the MSE converges
### def Solution2(x,t,N_hidden,eta, epoch,Minibach, W_hidden, W_out, b_hidden, b_out):
```
    cost_func_wh_grad = elementwise_grad(cost_function,2)
    cost_func_bh_grad = elementwise_grad(cost_function,4)
    cost_func_wo_grad = elementwise_grad(cost_function,3)
    cost_func_bo_grad = elementwise_grad(cost_function,5)
    activation_function = Sigmoid
    output_function = Identity
    MiniBachSize =int(np.size(x)/Minibach)
    for e in range(epoch):
        for j in range(Minibach):
            miniBach = np.random.randint(Minibach)
            miniBachMin, miniBachMax = MiniBachSize * miniBach,(MiniBachSize) * (miniBach+1)
            x_bach,t_bach = x[miniBachMin:miniBachMax],t[miniBachMin:miniBachMax]
            W_hidden -= eta*cost_func_wh_grad(x_bach,t_bach,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
            W_out -= eta*cost_func_wo_grad(x_bach,t_bach,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
            b_hidden -= eta*cost_func_bh_grad(x_bach,t_bach,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
            b_out -= eta*cost_func_bo_grad(x_bach,t_bach,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
      #  print("finished iteration number: ", e)
   
   
    return W_hidden, W_out, b_hidden, b_out
```
