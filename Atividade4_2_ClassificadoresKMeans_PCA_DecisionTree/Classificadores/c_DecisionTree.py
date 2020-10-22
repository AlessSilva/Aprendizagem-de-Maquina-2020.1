import numpy as np
import math


class DecisionTree( object ):

    def __init__( self, max_depth, min_size ):

        self.MAX_DEPTH = max_depth

        self.MIN_SIZE = min_size

        self.root = None

        self.X_train = None

        self.Y_train = None

        self.Y_predict = None

    def fit( self, X_train, Y_train ):

        self.X_train = np.array( X_train )

        self.Y_train = np.array( Y_train )

        self.root = self.__get_split( self.X_train, self.Y_train )

        self.__split( self.root, self.Y_train, 1)

    def predict( self, X_predict ):

        X_predict = np.array( X_predict )

        Y_predict = []

        for x in X_predict:

            Y_predict.append( self.__predict_tree( self.root, x ) )

        self.Y_predict = np.array(Y_predict).reshape(-1,1)

        return self.Y_predict
   

    def __predict_tree(self, node, x ):

        if x[ node['feature'] ] < node['value']:
            
            if isinstance( node['left_child'], dict ):
                
                return self.__predict_tree(node['left_child'], x)
            
            else:
            
                return node['left_child']
        
        else:
        
            if isinstance( node['right_child'], dict):
                
                return self.__predict_tree( node['right_child'], x)
            
            else:
                
                return node['right_child'] 

    def __split( self, node, Y, depth ):

        left_child, right_child = node["children"]

        del( node["children"] )

        if not left_child or not right_child:

            node['left_child'] = node['right_child'] = self.__to_terminal( left_child + right_child )
            
            return

        if depth >= self.MAX_DEPTH:
            
            node['left_child'] = self.__to_terminal( left_child )

            node['right_child'] = self.__to_terminal( right_child )

            return
   
        if len(left_child) <= self.MIN_SIZE:

            node['left_child'] = self.__to_terminal( left_child )

        else:

            X_ = np.array( [ elem['x'] for elem in left_child ] )

            Y_ = np.array( [ elem['y'] for elem in left_child ] )
 
            node['left_child'] = self.__get_split( X_, Y_ )

            self.__split( node['left_child'], Y_, depth+1 )

        if len(right_child) <= self.MIN_SIZE:

            node['right_child'] = self.__to_terminal( right_child )
        
        else:

            X_ = np.array( [ elem['x'] for elem in right_child ] )

            Y_ = np.array( [ elem['y'] for elem in right_child ] )
 
            node['right_child'] = self.__get_split( X_, Y_ )

            self.__split( node['right_child'], Y_, depth+1 )

    def __get_split( self, X, Y ):

        classes = np.unique( Y )

        best_feature = None
        best_value = float("inf")
        best_gini = float("inf")
        best_children = None

        for feature in range(X.shape[1]):

            for x1 in X:

                value = x1[feature]

                children = self.__test_split( feature, value, X, Y )

                gini = self.__gini_calculator( children, classes )

                if gini < best_gini:

                    best_feature = feature
                    best_value = value
                    best_gini = gini
                    best_children = children

        return { 'feature': best_feature,
                 'value': best_value,
                 'children': best_children }


    def __test_split( self, feature, value, X, Y ):

        left_child = []
        right_child = []

        for x1, y1 in zip(X, Y):

            if x1[ feature ] < value:

                left_child.append( { 'x' : x1, 'y' : y1 } )

            else:

                right_child.append( { 'x' : x1, 'y' : y1 } )

        return left_child, right_child


    def __gini_calculator( self, children, classes ):

        num_total = len(children[0]) + len(children[1])

        gini = 0

        for child in children:

            labels = np.array( [ elem['y'] for elem in child ] )
            num_nodes = len(labels)

            if num_nodes == 0:
                continue

            f_sum = 0

            for c in classes:

                num_class = len(labels[ labels == c ])
                f = (num_class / num_nodes) ** 2

                f_sum += f
            
            gini += (num_nodes/num_total) * (1 - f_sum)

        return gini

    def __to_terminal( self, child ):
        
        labels = np.array( [ elem['y'] for elem in child ] ).flatten()
        
        return np.bincount( labels ).argmax()