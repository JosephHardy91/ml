__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import numpy as np


class Closed_Least_Squares(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.weights = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        return self


z = lambda x: np.e * x[0] + np.pi * x[1]
w = lambda x: np.e * x[0] + np.pi * x[1]

X = np.random.rand(10, 2)
y = z(X.T)
y2 = w(X.T)

cls = Closed_Least_Squares()
cls.fit(X, y)

cls2 = Closed_Least_Squares()
cls2.fit(X, y2)

import matplotlib.pyplot as plt
plt.scatter(X,cls.weights*X)
plt.show()
