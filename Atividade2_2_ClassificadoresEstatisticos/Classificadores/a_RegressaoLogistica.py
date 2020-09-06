import numpy as np
import math


class RegressaoLogistica(object):

    def __init__(self):

        self.x_train = None
        self.y_train = None
        
        self.y_predict = None
        
        self.THRESHOLD = None

        self.CLASSES = []
        self.WEIGHTS_CLASSES = {}

        self.EPHOCS = None
        self.ALPHA = None
        self.LAMBD = None

    #Função que adiciona um coluna de uns aos dados (bias)
    def __addBias(self, x):

        bias = np.ones((x.shape[0], 1))

        return np.concatenate((bias, x), axis=1)

    #Função que inicializa os pesos
    def __initWeights(self):

        return np.zeros((self.x_train.shape[1]+1, 1))

    #Calcula a probabilidade de x
    def __calculatePx(self, x,  weights):

        exp = -np.dot(x, weights)

        px = 1 / ( 1 + math.e ** exp )

        return px

    #Retorna as classes de acordo com as probabilidades de x
    def __calculateY(self, px):

        y = [ 1 if p[0] >= self.THRESHOLD else 0 for p in px ]

        y = np.array( y ).reshape(-1,1)

        return y

    #Retorna os rótulos com 1 para instâncias na classe c ou 0 caso contrário
    def __createXC_and_YC(self, c):

        xc = self.x_train.copy()
        yc = [ 1 if yi[0] == c else 0 for yi in self.y_train.copy()]
        yc = np.array( yc ).reshape(-1,1)

        return xc, yc


    #Gradiente descendente
    def __gradientDescent(self, x, y ):

        x = self.__addBias(x)

        weights = self.__initWeights()

        n = x.shape[0]

        for _ in range(self.EPHOCS):

            px = self.__calculatePx(x, weights)

            y_predict = self.__calculateY(px)

            e = y - y_predict

            weights_grad = (np.sum(e*x, axis=0)/n).reshape(-1, 1)

            regularization = weights * (self.LAMBD/n)
            regularization[0] = 0 # regularização em w0 é 0 

            weights += self.ALPHA * (weights_grad - regularization)
        
        return weights

    #Treina o modelo
    def fit(self, X, Y, threshold=0.5, alpha=0.01, ephocs=100, lambd=1):

        self.x_train = np.array(X)

        self.y_train = np.array(Y)

        self.THRESHOLD =  threshold

        self.ALPHA = alpha

        self.EPHOCS = ephocs

        self.ALPHA = alpha

        self.LAMBD = lambd

        #Para cada classe
        for c in np.unique(self.y_train):

            self.CLASSES.append(c)
            
            xc, yc = self.__createXC_and_YC(c)

            self.WEIGHTS_CLASSES[ c ] = self.__gradientDescent(xc, yc)

        
    def predict(self, X):

        x_test = np.array(X)

        x_test = self.__addBias(x_test)

        self.y_predict = []

        #Para cada instância
        for x in x_test:

            prob = 0
            class_x = None

            #Para cada classe
            for c in self.CLASSES:

                px = self.__calculatePx( x, self.WEIGHTS_CLASSES[c] )

                if px > prob:
                    prob = px
                    class_x = c

            self.y_predict.append(class_x)
        
        self.y_predict = np.array( self.y_predict ).reshape(-1,1)

        return self.y_predict