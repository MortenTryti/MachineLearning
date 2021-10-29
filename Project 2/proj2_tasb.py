import autograd.numpy as np
#import numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from matplotlib import pyplot as plt



def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def neuralnet_init(x, f, network_shape):
    # Network_shape is the shape of the number of nodes per layer so fex [2,1,6,4,osv]
    # Moreover network shape is also everything AFTER the input layer
    # Input
    nlayers = len(network_shape)

    # Setting up the weights and the biases for each layer
    weights = []
    b = []
    xdim = x.shape[1]  # dimention
    weights.append(np.random.randn(xdim, network_shape[0]))  # initialize
    # filling up with the general layers and weights and biases between them
    for i in range(1, nlayers):
        weights.append(np.random.randn(network_shape[i - 1], network_shape[i]))

    for i in range(nlayers):
        b.append(np.random.randn(1, network_shape[i]))

    return weights, b


def feed_forward(x, f, weights, b):
    Z = []  # function input, zl in lecture notes
    A = []  # function output
    nlayers = len(b)
    a_h = np.copy(x)  # here we gather the data for the different layers as vectors

    # actually loop over the layers
    for i in range(nlayers-1):
        iweights = weights[i]  # setting the weights for the layer

        z_h = np.matmul(a_h, iweights) + b[i]
        Z.append(z_h)

        a_h = f(z_h)  # this ia a^l in mortens notes
        A.append(a_h)
    print(z_h.shape,weights[-1].shape)
    returnvalue = np.matmul(z_h, weights[-1]) + b[-1]  # if last in output change to a_h this is z^L in mortens notes
    return Z, A, returnvalue


def CostFunc(y_tilde, y, alpha, W):
    # Ensure np array
    s = 0
    for w in W:
        w = np.array(w)
        s += (w ** 2).flatten().sum()

    return 0.5 / (len(y_tilde)) * np.sum((y_tilde - y) ** 2) + alpha * s


def back_prop(x, y, f, weights, b):
    Zl, Al, return_value = feed_forward(x, f, weights, b)  # this is l not 1
    nlayers = len(Zl)
    alpha = 0
    # error in the output layer
    #print(Zl[-1])
    delta_jL = elementwise_grad(f, 0)(Zl[-1]) * elementwise_grad(CostFunc, 0)(Zl[-1], y, alpha, weights)
    #print(elementwise_grad(CostFunc, 0)(Zl[-1], y, alpha, weights).shape , elementwise_grad(f, 0)(Zl[-1]).shape)
    delta_jl_p1 = np.copy(delta_jL)
    # error in the hidden layer
    #print(delta_jl_p1.shape ,Al[-1].shape)
    weights_gradient = [delta_jl_p1 @ Al[-1]]
    bias_gradient = [delta_jl_p1]

    for l in range(nlayers-1,2):
        #print(delta_jl_p1.T.shape, Al[l-1].shape)
        #print(delta_jl_p1.shape, weights[l+1].shape)
        #print(elementwise_grad(f, 0)(Zl[l]).shape)
        delta_jl = delta_jl_p1 @ weights[l+1] * elementwise_grad(f, 0)(Zl[l])
        delta_jl_p1 = np.copy(delta_jl)

        weights_gradient.append(delta_jl_p1.T @ Al[l-1])
        bias_gradient.append(delta_jl_p1)
        # gradients for the output layer

    return weights_gradient, bias_gradient


netshape = [2,3,1]
y = 2
x = np.zeros((1,3))
w,b = neuralnet_init(x,sigmoid,netshape)
wg,bg = back_prop(x,y,sigmoid,w,b)
print(wg)
print(b)
print(bg)


