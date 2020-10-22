import numpy as np
import math
from Utils.featureScaling import standardization

class PCA( object ):

    #Inicaliza o modelo
    #  k_dimension == número de componentes
    def __init__( self, k_dimension ):
        
        self.K = k_dimension

        self.W = None

        self.values = None

        self.vectors = None

        self.explained_variance = None

        self.X_train = None


    #Treina e transforma os dados
    def fit_transform( self, X_train, to_standardize=False ):

        self.X_train = np.array( X_train )

        if to_standardize:

            self.X_train = standardization( self.X_train )

        self.D = self.X_train.shape[1]

        #Construir matriz de covariância
        matrix_cov =  np.dot(
            (self.X_train - self.X_train.mean(axis=0)).T ,
            (self.X_train - self.X_train.mean(axis=0))) / ( self.X_train.shape[0] - 1 )

        #Decompor em autovalores e autovetores
        self.values, self.vectors = np.linalg.eig( matrix_cov )

        #Salvando quanto da variância é preservado
        self.explained_variance = []

        for v in self.values:

            self.explained_variance.append( v / np.sum(self.values) )

        #Ordenar os autovalores
        indexes = np.argsort( self.values )[ -self.K: ][::-1]

        #Construir W
        self.W = self.vectors[ indexes ].T

        return self.X_train @ self.W