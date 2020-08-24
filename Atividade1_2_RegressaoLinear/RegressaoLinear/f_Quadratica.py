import numpy as np

from RegressaoLinear.c_MultivariadaMA import RegLinearMultivariadaMA


class RegLinearQuadratica(object):

    def __init__(self):

        self.x_train = None
        self.y_train = None
        self.y_predict = None
        self.weigths = None
        self.RegMultipla = RegLinearMultivariadaMA()

    def __addPower(self, x):

        return np.concatenate((x, x**2), axis=1)

    def fit(self, x, y):

        self.x_train = np.array(x)
        self.y_train = np.array(y)

        self.x_train = self.__addPower(self.x_train)

        self.RegMultipla.fit(x=self.x_train, y=self.y_train)

        self.weigths = self.RegMultipla.weigts

    def predict(self, x):

        x = np.array(x)

        x = self.__addPower(x)

        self.y_predict = self.RegMultipla.predict(x=x)

        return self.y_predict