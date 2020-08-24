import numpy as np
from Metricas.MSE import MSE


class RegLinearMultivariadaSGD(object):

    def __init__(self):

        self.x_train = None
        self.y_train = None
        self.y_predict = None
        self.weights = None
        self.ephocs = None
        self.alpha = None
        self.history = None

    def __addBias(self, x):

        bias = np.ones((x.shape[0], 1))

        return np.concatenate((bias, x), axis=1)

    def __initWeights(self):

        self.weights = np.zeros((self.x_train.shape[1], 1))

    def fit(self, x, y, alpha=0.001, ephocs=100):

        self.x_train = np.array(x)
        self.y_train = np.array(y)

        self.x_train = self.__addBias(self.x_train)

        self.__initWeights()

        self.ephocs = ephocs

        self.alpha = alpha

        n = self.x_train.shape[0]

        self.history = []

        for _ in range(self.ephocs):

            for i in range(n):

                y_predict = np.dot(self.x_train, self.weights)

                e = self.y_train[i] - y_predict[i]

                weights_grad = (e*self.x_train[i]).reshape(-1, 1)

                self.weights += self.alpha * weights_grad

            self.history.append(MSE(self.y_train, y_predict))

            np.random.shuffle(self.x_train)

    def predict(self, x):

        x = np.array(x)

        x = self.__addBias(x)

        self.y_predict = np.dot(x, self.weights)

        return self.y_predict
