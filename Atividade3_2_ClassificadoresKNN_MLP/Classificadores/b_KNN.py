import numpy as np

class KNN ( object ):

    #Inicializa o modelo
    def __init__( self, k, distance="euclidean" ):
        
        self.K = k
        
        self.DISTANCE = distance #distance="manhattan" ou distance="euclidean"
        
        self.X_train = None
        
        self.Y_train = None
        
        self.Y_predict = None

    #Treina o modelo
    def fit( self, X, Y ):
        
        self.X_train = np.array(X)

        self.Y_train = np.array(Y)

    #Calcula a distância entre duas instâncias
    def __distance_calculator( self, x1, x2 ):
        
        if self.DISTANCE == "manhattan":
            
            return np.sum( np.absolute( x1 - x2 ) )
        
        else:
            
            return np.sqrt( np.sum( ( x1 - x2 )**2 ) )

    #Retorna os indices dos k vizinhos mais próximos de x1
    def __find_k_neighborhoods( self, x1 ):
        
        distances = []
        
        for x2 in self.X_train:
            
            distance = self.__distance_calculator( x1, x2 )
            
            distances.append( distance )
            
        k_neighborhoods = np.argpartition( distances, self.K )[: self.K]

        return k_neighborhoods

    #Retorna a classe/label com maior ocorrência na vizinhança
    def __label_calculator( self, k_neighborhoods ):
    
        labels = self.Y_train[ k_neighborhoods ].flatten()
        
        y = np.bincount( labels ).argmax()
        
        return y

    #Prediz a classe/label dos dados de entrada
    def predict( self, X_predict ):
        
        X_predict = np.array(X_predict)
        
        Y_predict = []
        
        for x1 in X_predict:
            
            k_neighborhoods = self.__find_k_neighborhoods( x1 )
            
            y = self.__label_calculator( k_neighborhoods )
            
            Y_predict.append( y )
            
        self.Y_predict = np.array(Y_predict).reshape(-1,1)

        return self.Y_predict