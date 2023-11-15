import numpy as np
from src.Activations import softmax


def cross_entropy(y_pred, y_true):
    y_pred = softmax(y_pred)
    loss = -np.sum(y_true * np.log(y_pred))  # Added epsilon for numerical stability
    return loss


def cross_entropy_derivative(y_pred, y_true):
    # Assuming y_pred is the output of softmax
    return y_pred - y_true
