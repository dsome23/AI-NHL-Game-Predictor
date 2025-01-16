import random

import numpy as np
from numpy.ma.core import asarray
from scipy.spatial import distance
from scipy import stats
import math

class KNN:
    """
    Implementation of the k-nearest neighbors algorithm for classification.
    """
    def __init__(self, k):
        """
        Takes one parameter.  k is the number of nearest neighbors to use
        to predict the output variable's value for a query point.
        """
        self.k = k
        
    def fit(self, X, y):
        """
        Stores the reference points (X) and their known output values (y).
        """
        self.X = X
        self.y = y
        
    def predict_loop(self, X):
        """
        Predicts the output variable's values for the query points X using loops.
        """
        arr = []
        for i in range(len(X)):
            arr.append(self.get_closest(X[i]))
        return np.asarray(arr)

    def get_closest(self, X):
        """
        Gets the closest points and returns the expected y
        """
        arr = []
        for i in range(len(self.X)):
            arr.append([i,distance.euclidean(self.X[i], X)])
        arr.sort(key=lambda x: x[1])
        arr = arr[:self.k]
        weights = {}
        for i in range(len(arr)):
            key = self.y[arr[i][0]]
            if key not in weights:
                weights[key] = 0
            weights[key] += 1 / (arr[i][1] + 1e-9)
        return max(weights, key=weights.get)