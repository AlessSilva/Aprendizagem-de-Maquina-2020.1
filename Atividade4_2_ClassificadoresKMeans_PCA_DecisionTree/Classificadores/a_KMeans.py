import numpy as np
import math

class KMeans( object ):

    #Inicalizando o modelo
    def __init__( self, n_clusters=5, distance="euclidean", inertia_tool=0.00001 ):
        
        self.N_CLUSTERS = n_clusters

        self.DISTANCE = distance

        self.INERTIA_TOOL = inertia_tool

        self.inertia = None

        self.centroids = None

        self.X_train = None

        self.Y_train = None

        self.Y_predict = None

    #Treinando o modelo
    def fit( self, X, max_iteration=500, centroids_from="data" ):

        self.X_train = np.array( X )

        self.__initial_centroids( self.X_train, centroids_from )

        inertia = float("inf")

        for _ in range(max_iteration):

            self.Y_train = []

            for x1 in self.X_train:

                centroid_index = self.__find_centroid( x1 )

                self.Y_train.append( centroid_index  )

            new_inertia = self.__update_centroids( self.X_train, self.Y_train )

            if abs(new_inertia - inertia) <= self.INERTIA_TOOL:

                break

            inertia = new_inertia

        self.inertia = inertia


    #Calcula os centroides iniciais
    def __initial_centroids( self, X, _from="data" ):

        if _from == "random":

            self.centroids = np.random.uniform( low=X.min(axis=0), high=X.max(axis=0), size=( self.N_CLUSTERS, X.shape[1] ) )

        elif _from == "data":

            indices = np.random.randint( low=0, high=X.shape[0], size=self.N_CLUSTERS)

            self.centroids = X[ indices ]

        
    #Calcula a distância entre dois pontos
    def __distance_calculator( self, x1, x2 ):
        
        if self.DISTANCE == "manhattan":
            
            return np.sum( np.absolute( x1 - x2 ) )
        
        elif self.DISTANCE == "euclidean":
            
            return np.sqrt( np.sum( ( x1 - x2 )**2 ) )


    #Encontra o centroid mais próximo daquela instância
    def __find_centroid( self, x1 ):

        distances = []

        for c in self.centroids:

            distance = self.__distance_calculator( x1, c )

            distances.append( distance ) 

        centroid_index = np.argmin( distances )

        return centroid_index


    #Atualiza o valor dos centroids
    def __update_centroids( self, X, Y ):

        inertia = 0

        for i,c in enumerate( self.centroids ):

            indexes = np.where( np.array(Y) == i )[0]

            if indexes.shape[0] > 0:


                mean = X[ indexes ].mean( axis=0 )

                self.centroids[i] = mean

                for x1 in X[ indexes ]:

                    inertia += np.square( self.__distance_calculator( x1, self.centroids[i] ) ) 

        return inertia

    
    #Etapa de predição
    def predict( self, X_predict ):

        X_predict = np.array( X_predict )

        Y_predict = []

        for x1 in X_predict:

                centroid_index = self.__find_centroid( x1 )

                Y_predict.append( centroid_index )

        self.Y_predict = np.array(Y_predict).reshape(-1,1)

        return self.Y_predict

    # def mean_distance_centroids( self ):

    #     sum = 0

    #     for i,c in enumerate( self.centroids ):

    #         indexes = np.where( np.array(self.Y_train) == i )

    #         for x1 in self.X_train[ indexes ]:

    #             sum += self.__distance_calculator( x1, self.centroids[i] ) 

    #     mean = sum / self.X_train.shape[0]
        
    #     return mean