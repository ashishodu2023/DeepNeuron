import numpy as np

# activation function and its derivative
def SIGMOID(x):
    return 1 / (1 + np.exp(-x))

def SIGMOID_PRIME(x):
    return np.exp(-x) / (1 + np.exp(-x))**2

def TANH(x):
    return np.tanh(x)

def TANH_PRIME(x):
    return 1 - np.tanh(x)**2

def RELU(x):
    return np.maximum(x, 0)

def RELU_PRIME(x):
    return np.array(x >= 0).astype('int')