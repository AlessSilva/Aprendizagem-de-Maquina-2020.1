import numpy as np

def RSS( y_true, y_predict ):
    return np.sum( np.power((y_true-y_predict),2) )