__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from iris_loadin import *

import matplotlib.pyplot as plt


# information gain is defined as follows:
# IG(Dp,f) = I(Dp) - sum((Nj/Np)*I(Dj) for j in range(1,m))
# for binary classifications
# IG(Dp,a) = I(Dp) - (Nleft/Np)*I(Dleft) - (Nright/Np)*I(Dright)
# where Dp and Dj are the datasets of the parent and jth child node,
# I is the impurity function/heuristic, f is the feature being tested against, and N is the number of samples
# in the child and parent nodes


# there are three kinds of impurity heuristics: entropy, gini, and classification error
# impurity errors are maximized if the proportions of each class in the node data are
# equal, i.e. impurity is maximized and information gain is minimized
# Entropy:
#   Ih(t) = -sum(p(i|t)*log2(p(i|t))  for i in range(1,c))
# Gini Index:
#   Ig(t) = 1 - sum(p(i|t)^2 for i in range(1,c))
# Classification Error:
#   Ie(t) = 1 - max(p(i|t) for i in range(1,c))

def gini(p):
    return (p * (1 - p)) + ((1 - p) * (1 - (1 - p)))


def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2((1 - p))


def error(p):
    return 1 - np.max([p, 1 - p])


def compare_impurity_heuristics():
    x = np.arange(0.0, 1.0, 0.01)
    ent = [entropy(p) if p != 0 else None for p in x]
    sc_ent = [e * 0.5 if e else None for e in ent]
    err = [error(i) for i in x]
    fig = plt.figure()
    ax = plt.subplot(111)
    for i, lab, ls, c in zip([ent, sc_ent, gini(x), err],
                             ['Entropy', 'Entropy (Scaled)', 'Gini Impurity', 'Misclassification Error'],
                             ['-', '-', '--', '-.'],
                             ['black', 'lightgray', 'red', 'green', 'cyan']):
        line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=3, fancybox=True, shadow=False)
    ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
    plt.ylim([0, 1.1])
    plt.xlabel('p(i=1)')
    plt.ylabel('Impurity Index')
    plt.show()


def classify_iris():
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    tree.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
    plt.xlabel('petal length[cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()
    from sklearn.tree import export_graphviz
    export_graphviz(tree, out_file='tree.dot', feature_names=['petal length', 'petal width'])
    os.system('dot -Tpng tree.dot -o tree.png')
    os.system('explorer tree.png')


def classify_iris_random_forest():
    forest = RandomForestClassifier(criterion='entropy',
                                    n_estimators=10,
                                    random_state=1,
                                    n_jobs=2)

    forest.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc='upper left')
    plt.show()

classify_iris_random_forest()
