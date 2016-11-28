__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron,plot_decision_regions

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data',
                 header=None)

y = df.iloc[50:150, 4].values
# setosa,virginica,versicolor
y = np.where(y == 'Iris-versicolor', -1, 1)
X = df.iloc[50:150, [0, 2]].values

# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
#
# plt.xlabel('petal length [cm]')
# plt.ylabel('sepal length [cm]')
# plt.legend(loc='upper left')
# plt.show()

ppn = Perceptron(eta=0.1, n_iter=100,progress_bar=False)
ppn.fit(X, y)
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of misclassifications')
# plt.show()



plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('petal length [cm]')
plt.ylabel('sepal length [cm]')
plt.legend(loc='upper left')
plt.show()
