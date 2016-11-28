__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import AdalineGD, AdalineSGD,plot_decision_regions
import matplotlib.pyplot as plt

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data',
                 header=None)

y = df.iloc[0:100, 4].values
# setosa,virginica,versicolor
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values


def show_bad_etas(X, y):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(SSE)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('SSE')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    plt.show()


def iris_batch_descent(X, y):
    X_std = np.copy(X)
    for k in range(2):
        X_std[:, k] = (X[:, k] - X[:, k].mean()) / X[:, k].std()

    ada = AdalineGD(n_iter=15, eta=0.01)
    ada.fit(X_std, y)

    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('SSE')
    plt.show()

def iris_stochastic_descent(X,y):
    X_std = np.copy(X)
    for k in range(2):
        X_std[:, k] = (X[:, k] - X[:, k].mean()) / X[:, k].std()

    ada = AdalineSGD(n_iter=15,eta=0.01,random_state=1)
    ada.fit(X_std,y)
    plot_decision_regions(X_std,y,classifier=ada)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.show()

iris_stochastic_descent(X,y)