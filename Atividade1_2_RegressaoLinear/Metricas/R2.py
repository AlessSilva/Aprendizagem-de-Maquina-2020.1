import numpy as np
from Metricas.RSS import RSS


def R2(y_true, y_predict):
    y_mean = np.mean(y_true)
    tss = np.sum(np.power((y_true-y_mean), 2))
    rss = RSS(y_true, y_predict)
    return 1 - (rss/tss)
