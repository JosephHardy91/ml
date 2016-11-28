__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from perceptron import plot_decision_regions


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def plot_test_sigmoid(s):
    z = np.arange(-s, s, 0.001)
    phi_z = sigmoid(z)
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
    plt.axhline(y=0.5, ls='dotted', color='k')
    plt.yticks([0.0, 0.5, 1.0])
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')
    plt.show()


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # pedal length, pedal width
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

def basic_logistic():

    lr = LogisticRegression(C=1000.0, random_state=0)
    lr.fit(X_train_std, y_train)

    plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()

    y_pred = lr.predict(X_test_std)
    print 'Misclassified samples:\t%d of %d\nMisclassification rate:\t%f%%' % (
        (y_test != y_pred).sum(), X_test_std.shape[0], 100 * ((y_test != y_pred).sum() / float(X_test_std.shape[0])))

    print 'Accuracy:%.2f' % accuracy_score(y_test, y_pred)

def regularized_logistic():
    weights,params = [],[]
    for c in np.arange(-5,5):
        lr = LogisticRegression(C=10**c,random_state=0)
        lr.fit(X_train_std,y_train)
        #print lr.coef_
        #print lr.coef_[1]
        weights.append(lr.coef_[1])
        params.append(10**c)

    weights = np.array(weights)
    plt.plot(params,weights[:,0],label='petal length')
    plt.plot(params,weights[:,1],label='petal width',linestyle='--')
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.legend(loc='upper left')
    plt.xscale('log')
    plt.show()

regularized_logistic()