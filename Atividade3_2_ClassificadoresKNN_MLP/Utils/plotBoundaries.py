import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_boundaries(x, y, clf):
    
    y = y.reshape(1,-1)[0]

    x1_min, x1_max = x[:, 0].min() - .5, x[:, 0].max() + .5

    x2_min, x2_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    
    h = .02 
    
    x1x, x2x = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    Z = clf.predict(np.c_[x1x.ravel(), x2x.ravel()])

    Z = Z.reshape(1,-1)[0]
    Z = Z.reshape(x1x.shape)
    
    plt.figure(1, figsize=(6, 6))
    plt.pcolormesh(x1x, x2x, Z, cmap=plt.cm.Spectral)
    
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.Spectral, s=60)
    plt.xlabel('Feature 0',fontsize=12)
    plt.ylabel('Feature 1',fontsize=12)

    plt.xlim(x1x.min(), x1x.max())
    plt.ylim(x2x.min(), x2x.max())
    plt.xticks(())
    plt.yticks(())
    
    plt.show()