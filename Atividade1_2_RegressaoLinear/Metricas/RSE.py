import numpy as np
from Metricas.RSS import RSS


def RSE(y_true, y_predict):
    n = y_true.shape[0]
    rss = RSS(y_true, y_predict)
    return np.power((rss/(n-2)), 0.5)
