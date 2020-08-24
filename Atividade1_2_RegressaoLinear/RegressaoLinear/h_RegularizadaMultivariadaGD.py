import numpy as np


class RegLinearRegularizadoMultivariadaGD(object):

    def __init__(self):

        self.x_train = None
        self.y_train = None
        self.y_predict = None
        self.weights = None
        self.ephocs = None
        self.alpha = None
        self.lambd = None  # <------------------------------------------------------- Regularização

    def __addBias(self, x):

        bias = np.ones((x.shape[0], 1))

        return np.concatenate((bias, x), axis=1)

    def __initWeights(self):

        self.weights = np.zeros((self.x_train.shape[1], 1))

    def fit(self, x, y, alpha=0.001, ephocs=100, lambd=1):

        self.x_train = np.array(x)
        self.y_train = np.array(y)

        self.x_train = self.__addBias(self.x_train)

        self.__initWeights()

        self.ephocs = ephocs

        self.alpha = alpha

        self.lambd = lambd

        n = self.x_train.shape[0]

        for _ in range(self.ephocs):

            y_predict = np.dot(self.x_train, self.weights)

            e = self.y_train - y_predict

            weights_grad = (np.sum(e*self.x_train, axis=0)/n).reshape(-1, 1)

            # ------------------------ Regularização ----------------------------------
            regularization = self.weights * (self.lambd/n)
            regularization[0] = 0 # regularização em w0 é 0 

            self.weights += self.alpha * (weights_grad - regularization)
            # <----------------------------------------------------------------

    def predict(self, x):

        x = np.array(x)

        x = self.__addBias(x)

        self.y_predict = np.dot(x, self.weights)

        return self.y_predict
