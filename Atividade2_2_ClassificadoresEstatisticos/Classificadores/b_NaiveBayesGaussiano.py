import numpy as np
import math

class NaiveBayesGaussiano(object):
    
    def __init__(self):

        self.x_train = None
        self.y_train = None

        self.y_predict = None

        self.N = 0
        self.M = 0

        self.CLASSES = []
        self.PROB_CLASS = {}
        self.MEAN_CLASS = {}
        self.MATRIX_CLASS = {}
    
    def __calculateProbClass(self, c):

        #Descobre os indices dos elementos com aquela classe
        index_c = np.where( self.y_train == c )[0]
            
        #Quantidade de elementos da classe c
        nc = len(index_c)

        #Elementos da classe c
        xc = self.x_train[index_c]
            
        #Calculando probabilidade da classe
        prob = nc/self.N

        return prob

    def __calculateMeanClass(self, c):

        #Descobre os indices dos elementos com aquela classe
        index_c = np.where( self.y_train == c )[0]

        #Elementos da classe c
        xc = self.x_train[index_c]

        #Calculando a média da classe
        return np.mean(xc,axis=0)

    def __calculateMatrixClass(self, c):

        #Descobre os indices dos elementos com aquela classe
        index_c = np.where( self.y_train == c )[0]

        #Elementos da classe c
        xc = self.x_train[index_c]

        #Calculando matriz de covariância
        n = xc.shape[0]
        matrix_cov =  np.dot((xc - self.MEAN_CLASS[c]).T , (xc - self.MEAN_CLASS[c])) / ( n - 1 ) #np.cov(xc.T)

        #Pegando os valores da diagonal
        diagonal_values = list(matrix_cov.diagonal())

        #Criando uma matriz com zeros
        matrix_nb = np.zeros(matrix_cov.shape)
            
        #Adicionando a matriz com zeros os valores da diagonal
        np.fill_diagonal(matrix_nb,diagonal_values)

        return matrix_nb


    def fit(self, X, Y):
        
        self.x_train = np.array(X)
        self.y_train = np.array(Y)

        self.N = len(self.y_train)
        
        #Para cada classe
        for c in np.unique(self.y_train):
            
            #Adiciona a classe ao array de classes
            self.CLASSES.append(c)

            #Adicionando a probabilidade daquela classe ao dicionário de probabilidades
            self.PROB_CLASS[c] = self.__calculateProbClass(c)
            
            #Adicionando a média daquela classe ao dicionário de médias
            self.MEAN_CLASS[c] = self.__calculateMeanClass(c)
            
            #Adicionando a matriz daquela classe ao dicionário de matrizes
            self.MATRIX_CLASS[c] = self.__calculateMatrixClass(c)
        
    def __calculateProb_x_c( self, x, c):

        #--------------- Calculando a probabilidade de x ser da classe c -----

        pxc_1 = np.linalg.det(self.MATRIX_CLASS[c])
        pxc_1 = np.sqrt(pxc_1) * ((2*math.pi)**(self.M/2))
        pxc_1 = 1/pxc_1
                
        pxc_2 = np.dot((x-self.MEAN_CLASS[c]).T,np.linalg.inv(self.MATRIX_CLASS[c]))
        pxc_2 = (-0.5) * np.dot(pxc_2,(x-self.MEAN_CLASS[c]))
        pxc_2 = math.exp(pxc_2)
                
        pxc = pxc_1*pxc_2
                
        pcx = pxc*self.PROB_CLASS[c]

        return pcx

    def predict(self, X):
        
        x_test = np.array(X)

        self.y_predict = []

        #Descobrindo o número de atributos
        self.M = x_test.shape[1]
        
        #Para cada instância de dados
        for x in x_test:
            
            prob = 0
            class_x = None
            
            #Para cada classe
            for c in self.CLASSES:

                #--------------- Calculando a probabilidade de x ser da classe c -----
                pcx = self.__calculateProb_x_c(x, c)
                
                #Salvando a maior probabilidade encontrada até o momento
                if pcx > prob:
                    prob = pcx
                    class_x = c
                
            self.y_predict.append(class_x)

        self.y_predict = np.array(self.y_predict).reshape(-1,1)
        
        return self.y_predict