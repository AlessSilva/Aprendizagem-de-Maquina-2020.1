import numpy as np

def normalization ( x_train, x_test ):
    
    x_min = x_train.min(axis=0)
    x_max = x_train.max(axis=0)
    
    x_train = (x_train - x_min) / ( x_max - x_min )
    x_test = (x_test - x_min) / ( x_max - x_min )
    
    return x_train, x_test

def mean_normalization ( x_train, x_test ):
    
    x_mean = x_train.mean(axis=0)
    x_min = x_train.min(axis=0)
    x_max = x_train.max(axis=0)
    
    x_train = (x_train - x_mean) / ( x_max - x_min )
    x_test = (x_test - x_mean) / ( x_max - x_min )
    
    return x_train, x_test

def padronization ( x_train, x_test ):
    
    x_mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    
    x_train = (x_train - x_mean) / std
    x_test = (x_test - x_mean) / std
    