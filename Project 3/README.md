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
```
    return np.exp(y)/(1+np.exp(y))
```

### def Identity(y):
Applies the identity function to input y, so nothing happens.
```
    return y
```

### RELU(y)
Calculates the function values of the RELU activation function.
```
    return np.maximum(0,y)
```

### lexyRelu(y)
Calculates the function values of the Leaky-RELU activation function.
```
    return np.maximum(0.01*y,y)
```

### Set_weights_and_bias(n_in,n_hidden,n_out)
Takes number of input nodes, number of hidden nodes and number of outputnodes.
Returns Weights for hidden layer, weights for the output layer, the output for the hidden layer(s) and the output for the output layer.
```
#### Setting hiden weights
    W_hidden = 0.1*np.random.randn(n_in, n_hidden)
    b_hidden = np.zeros(n_hidden) +0.01
#### setting output weights
    W_out = 0.1*np.random.randn(n_hidden, n_out)
    b_out = np.zeros(n_out) +0.01
    return W_hidden, W_out, b_hidden, b_out
```


### feed_forward_train(X,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
(From lecture notes)

(Used in order to use our own backpropagation from project 2)
Takes as input the designmatrix, X, the weights for the output layer, the output for the hidden layer(s) and the output for the output layer, an activation function and an output function. Returns the output of the activation function in the hidden layer(s), the output of the output function on the output layer, the output from the hidden layer(s) and the output from the output layer.
```
#### Hidden attac
    z_h = X@W_hidden + b_hidden
    a_h = activation_function(z_h)
#### output attac
    z_o = a_h@W_out + b_out
    a_o = output_function(z_o)
    return a_h, a_o, z_h,z_o

```

### feed_forward(X,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
Does the same as feed_forward_train(), but only returns the output of the output function on the output layer as a scalar.
```
#### Hidden attac
    z_h = X@W_hidden + b_hidden
    a_h = activation_function(z_h)
#### output attac
    z_o = a_h@W_out + b_out
    a_o = output_function(z_o)
    return a_o[0]
```

### feed_forward_eigen(X,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
Does the same as feed_forward() but returns a_o as a reshaped vector.
```
#### Hidden attac

    z_h = X@W_hidden + b_hidden
    a_h = activation_function(z_h)
#### output attac
    z_o = a_h@W_out + b_out
    a_o = output_function(z_o)
    a_o = a_o.reshape(6,1)
    return a_o
```

### u(x)
Returns the function value of sin(pi*x).
```
    return np.sin(np.pi*x)
```

### g_trial(X,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
Uses feed_forward_eigen() and X to return a guess of trial function.
```
x,t = X[0],X[1]
    return (1-t)*u(x) + x*(1-x)*t*feed_forward_eigen(X,W_hidden, W_out, b_hidden, b_out,activation_function, output_function)
```


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

### def solution(x,t,N_hidden,eta, activation_function ,output_function, epoch,Minibach):
```

    # from notes, altered to fit our setup
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
