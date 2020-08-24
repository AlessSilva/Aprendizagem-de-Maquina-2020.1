import numpy as np


def MAE(y_true, y_predict):
    n = y_true.shape[0]
    return np.sum(abs(y_true-y_predict)) / n
