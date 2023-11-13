import numpy as np

# loss function and its derivative
def MSE(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def MSE_PRIME(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

# https://stackoverflow.com/questions/67615051/implementing-binary-cross-entropy-loss-gives-different-answer-than-tensorflows
def  BINARY_CROSS_ENTROPY(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)


# https://www.mldawn.com/binary-classification-from-scratch-using-numpy/
def BINARY_CROSS_ENTROPY_PRIME(y_true, y_pred):
    if y_true == 1:
        return -1 / y_pred
    else:
        return 1 / (1 - y_pred)