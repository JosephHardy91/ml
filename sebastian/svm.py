__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os

# SVMs attempt to maximize the margin
# the margin is the distance between the separating hyperplane and
# the training samples closest to this hyperplane
# (JH Note) - this seems to allow it to automatically avoid overfitting
# https://epub-imgs.scribdassets.com/35adi63ssg4rzd0i/images/image-YH3QRA9Z.jpg
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score
from iris_loadin import *
from sklearn.svm import SVC
from perceptron import plot_decision_regions


svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
print accuracy_score(y_test, svm.predict(X_test_std))
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
