import numpy as np


# activation function and its derivative

def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return np.array(x >= 0).astype('int')


def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(self, x):
    return x * (1 - x)


def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=0))
    return exp_values / np.sum(exp_values, axis=0)


def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)
