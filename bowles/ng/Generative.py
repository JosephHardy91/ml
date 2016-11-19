__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import numpy as np
import random
import matplotlib.pyplot as plt

def Gaussian_Discriminant(data):
    m = len(data)
    phi = sum(data[i][1] for i in range(m) if data[i][1] == 1)
    mus = [sum(data[i][0] for i in range(m) if data[i][1] == t) / sum(1 for i in range(m) if data[i][1] == t) for t in
           [0, 1]]
    print phi, mus
    #three ways to argmax
    #(P(x|y)P(y))/P(x)
    #P(x|y)P(y)
    #P(x|y) if P(y) is uniform

if __name__ == "__main__":
    data = []
    for _ in range(100):
        t = random.randint(0, 1)
        data.append([random.random() * (t + 1), t])
    data = np.asarray(data)
    # data = np.array([[random.random(), random.randint(0, 1)] for _ in range(100)])
    Gaussian_Discriminant(data)
