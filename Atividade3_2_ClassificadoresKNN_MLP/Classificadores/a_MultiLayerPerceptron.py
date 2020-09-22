import numpy as np
import math

class MultiLayerPerceptron( object ):

    ##Incia o modelo
    def __init__( self, input_layer_size=2, hidden_layer_size=4 ):

        self.X_train = None

        self.Y_train = None

        self.Y_predict = None

        self.W = None

        self.M = None

        self.INPUT_LAYER_SIZE = input_layer_size

        self.HIDDEN_LAYER_SIZE = hidden_layer_size

        self.OUTPUT_LAYER_SIZE = 1

        self.EPHOCS = None

        self.ALPHA = None

    ## Adiciona o bias/coluna de -1's
    def __addBias( self, X ):

        bias = np.ones( ( X.shape[0], 1 ) ) * -1

        return np.concatenate( ( bias, X ), axis = 1 )

    ## Inicializa os pesos W e M
    def __initWeights( self ):

        self.W = np.ones( ( self.INPUT_LAYER_SIZE + 1, self.HIDDEN_LAYER_SIZE ) )

        self.M = np.ones( ( self.HIDDEN_LAYER_SIZE + 1, self.OUTPUT_LAYER_SIZE ) )

    ##Função sig:3 / sigmoid
    def __sigmoid( self, x ):

        return 1 / ( 1 + math.e ** ( - x ) )

    ##Derivada da função sig:3 /sigmoid
    def __derivative_sigmoid( self, x ):

        fx = self.__sigmoid( x ) 

        return  fx * ( 1 - fx )

    ##Sentido direto
    def __fowardpropagation( self ):

        Ui = np.dot( self.X_train, self.W )

        Zi = self.__sigmoid( Ui )
        
        Uk = np.dot( self.__addBias( Zi ) , self.M )
        
        Y_predict = self.__sigmoid( Uk )

        Ek = self.Y_train - Y_predict

        return Y_predict, Ek, Ui, Zi, Uk

    ##Sentido inverso
    def __backwardpropagation( self, Ek, Ui, Zi, Uk ):

        delta_k = Ek * self.__derivative_sigmoid( Uk )

        self.M += self.ALPHA * ( np.dot(  self.__addBias(Zi).T, delta_k ) )

        delta_i = self.__derivative_sigmoid( Ui ) * ( np.dot( delta_k, np.delete( self.M, 0, 0 ).T ) ) 

        self.W += self.ALPHA * ( np.dot( self.X_train.T, delta_i ) )

    #Treina o modelo
    def fit( self, X, Y, alpha=0.1 ,ephocs=100):

        X = np.array( X )

        self.X_train = self.__addBias( X )

        self.Y_train = np.array( Y )

        self.ALPHA = alpha

        self.EPHOCS = ephocs

        self.__initWeights()

        for _ in range( self.EPHOCS ):

            Y_predict, Ek, Ui, Zi, Uk = self.__fowardpropagation()

            self.__backwardpropagation( Ek, Ui, Zi, Uk )

    #Prediz a classe/label dos dados de entrada
    def predict( self, X_predict ):

        X_predict = np.array( X_predict )

        X_predict = self.__addBias( X_predict )

        Ui = np.dot( X_predict, self.W )

        Zi = self.__sigmoid( Ui )
        
        Uk = np.dot( self.__addBias( Zi ) , self.M )
        
        Y_predict = self.__sigmoid( Uk )

        Y_predict = np.array([ 0 if y < 0.5 else 1 for y in Y_predict.flatten() ])
        
        self.Y_predict = Y_predict.reshape( -1, 1 )

        return self.Y_predict