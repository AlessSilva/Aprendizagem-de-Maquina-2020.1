import numpy as np

def MSE(y_true, y_predict):
    n = y_true.shape[0]
    return np.sum(np.power(y_true-y_predict, 2)) / n