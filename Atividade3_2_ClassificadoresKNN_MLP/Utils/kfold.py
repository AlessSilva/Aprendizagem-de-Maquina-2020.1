import numpy as np

from Utils import accuracy

def kfold( X, Y, k, clf ):

    X = np.array( X )
    Y = np.array( Y ).reshape( -1, 1 )

    indexes = np.arange( X.shape[0] )

    slices = []

    aux1 =0
    aux2 = int( X.shape[0] / k )
    
    for i in range(k):

        if i == k-1:
            _slice = np.isin( indexes, indexes[ aux1 : ] )
        else:
            _slice = np.isin( indexes, indexes[ aux1: ( aux1 + aux2 ) ] )
        
        slices.append(_slice)
        
        aux1 += aux2

    accuracies = []

    for s in slices:
        
        x_train = X[ ~s ] 
        y_train = Y[ ~s ]

        x_test = X[ s ]
        y_test = Y[ s ]

        clf.fit( x_train, y_train )

        y_predict = clf.predict( x_test )        

        accuracies.append(accuracy( y_test, y_predict ))

    return np.mean(accuracies)

        