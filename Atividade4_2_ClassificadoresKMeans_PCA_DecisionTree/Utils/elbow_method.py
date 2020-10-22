import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Classificadores import KMeans

def elbow_method( X, n=10 ):

    inertias = np.zeros( n )

    for k in range( n ):

        model = KMeans( n_clusters=k+1 )
        model.fit( X )
        
        inertias[ k ] = model.inertia

    plt.figure( figsize=(10,7) )

    plt.plot( range(1,n+1), inertias )
    plt.xlabel("k")
    plt.ylabel("Inertia")

    plt.show()