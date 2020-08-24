import numpy as np


class RegLinearUnivariadaGD(object):

    def __init__(self):

        self.x_train = None
        self.y_train = None
        self.y_predict = None
        self.weight0 = None
        self.weight1 = None
        self.alpha = None
        self.ephocs = None

    def __initWeights(self):

        self.weight0 = 0
        self.weight1 = 0

    def fit(self, x, y, alpha=0.01, ephocs=100):

        self.x_train = np.array(x)

        self.y_train = np.array(x)

        self.__initWeights()

        self.alpha = alpha

        self.ephocs = ephocs

        n = self.x_train.shape[0]

        for _ in range(self.ephocs):

            y_predict = self.weight0 + self.weight1 * self.x_train

            e = self.y_train - y_predict

            weight0_grad = np.sum(e)/n

            weight1_grad = np.sum(e * self.x_train)/n

            self.weight0 += self.alpha * weight0_grad

            self.weight1 += self.alpha * weight1_grad

    def predict(self, x):

        x = np.array(x)

        self.y_predict = self.weight0 + self.weight1 * x

        return self.y_predict
