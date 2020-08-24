import numpy as np


class RegLinearMultivariadaMA(object):

    def __init__(self):

        self.x_train = None
        self.y_train = None
        self.y_predict = None
        self.weigts = None

    def __addBias(self, x):

        bias = np.ones((x.shape[0], 1))

        return np.concatenate((bias, x), axis=1)

    def fit(self, x, y):

        self.x_train = np.array(x)
        self.y_train = np.array(y)

        self.x_train = self.__addBias(self.x_train)

        self.weigts = self.x_train.T.dot(self.x_train)
        self.weigts = np.linalg.inv(self.weigts)
        self.weigts = self.weigts.dot(self.x_train.T)
        self.weigts = self.weigts.dot(self.y_train)

    def predict(self, x):

        x = np.array(x)

        x = self.__addBias(x)

        self.y_predict = np.dot(x, self.weigts)

        return self.y_predict
