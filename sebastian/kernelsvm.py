__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from xor import X_xor, y_xor
from perceptron import plot_decision_regions
from iris_loadin import *


# try to separate the xor data using hyperplane kernels

# rbf is the radial basis function kernel where the
# kernel function is k(xi,xj) = exp(-gamma*l2(xi-xj))
# and gamma is 1/(2*std**2)[which is freely edited depending on a perceived distribution]
# another way to think about the gamma parameter is as a cut-off parameter for the Gaussian sphere,
# such that the larger gamma gets, the more influence each training sample has on the overall decision boundary
# (decreasing gamma creates a softer[less firm] and more generalized decision boundary)
def xorSVM(y):
    svm = SVC(kernel='rbf', random_state=0, gamma=y, C=10.0)
    svm.fit(X_xor, y_xor)
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.show()


def irisSVM(y):
    svm = SVC(kernel='rbf', random_state=0, gamma=y, C=1.0)
    svm.fit(X_train_std, y_train)
    print accuracy_score(y_test, svm.predict(X_test_std))
    plot_iris_decision_regions_with_classifier(svm)


# higher values of gamma create more accurate but less well generalized divisions
# xorSVM(0.2)
# xorSVM(100.0)
# irisSVM(0.2)
# irisSVM(100.0)
