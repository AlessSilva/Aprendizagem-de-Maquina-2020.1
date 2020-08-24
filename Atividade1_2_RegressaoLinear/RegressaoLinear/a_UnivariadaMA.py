import numpy as np


class RegLinearUnivariadaMA(object):

    def __init__(self):

        self.x_train = None
        self.y_train = None
        self.y_predict = None
        self.weight0 = None
        self.weight1 = None

    def fit(self, x, y):

        self.x_train = np.array(x)
        self.y_train = np.array(y)

        x_mean = self.x_train.mean()
        y_mean = self.y_train.mean()

        self.weight1 = np.sum(
            ((self.x_train - x_mean)*(self.y_train - y_mean)))
        self.weight1 /= np.sum((self.x_train - x_mean)**2)

        self.weight0 = y_mean - (self.weight1 * x_mean)

    def predict(self, x):

        x = np.array(x)

        self.y_predict = self.weight0 + self.weight1 * x

        return self.y_predict
