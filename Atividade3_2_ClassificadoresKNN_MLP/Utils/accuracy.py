import numpy as np

def accuracy( y_true, y_predict ):
    
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)

    hits = (y_true == y_predict).sum()
    
    total = y_true.shape[0]
    
    return hits/total