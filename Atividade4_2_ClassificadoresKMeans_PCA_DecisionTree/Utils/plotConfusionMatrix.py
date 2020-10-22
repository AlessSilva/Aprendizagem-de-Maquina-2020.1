import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(x, y, clf):
    
    y = y.reshape(1,-1)[0]

    y_predict = clf.predict(x)
    y_predict = y_predict.reshape(1,-1)[0]

    matrix = pd.crosstab(y,y_predict)
    
    l = len(matrix.columns)
    
    plt.figure(figsize=(7,5))
    plt.matshow(matrix, cmap=plt.cm.Blues,fignum=1)
    plt.colorbar()
    
    tick_marks = np.arange(l)
    plt.xticks(tick_marks, matrix.columns,fontsize=12,rotation=-45,color="blue")
    plt.yticks(tick_marks, matrix.index,fontsize=12,rotation=-45,color="blue")
    
    for (i, j), z in np.ndenumerate(matrix):
        plt.text(j, i, z, ha='center', va='center',fontsize=12)
    
    plt.xlabel("Predicted Label",fontsize=12)
    plt.ylabel("True Label",fontsize=12)

    plt.show()