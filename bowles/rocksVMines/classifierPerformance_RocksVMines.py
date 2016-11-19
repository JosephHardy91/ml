__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot
import urllib2, numpy, random
from sklearn import datasets, linear_model
from sklearn.metrics import roc_curve, auc
import pylab as pl


def confusionMatrix(predicted, actual, threshold):
    if len(predicted) != len(actual): return -1
    tp, fp, tn, fn = [0.0 for _ in range(4)]
    for i in range(len(actual)):
        if actual[i] > 0.5:
            if predicted[i] > threshold:
                tp += 1.0
            else:
                fn += 1.0
        else:
            if predicted[i] < threshold:
                tn += 1.0
            else:
                fp += 1.0
    rtn = (tp, fn, fp, tn)
    return rtn


target_url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
)

data = urllib2.urlopen(target_url)

# arrange data into list for labels and list of lists for attributes
xList = []
labels = []
for line in data:
    # split on comma
    row = line.strip().split(",")
    # assign label 1.0 for "M" and 0.0 for "R"
    if row[-1] == 'M':
        labels.append(1.0)
    else:
        labels.append(0.0)
    # remove label from row
    row.pop()
    # convert row to floats
    floatRow = [float(num) for num in row]
    xList.append(floatRow)

# divide attribute matrix and label vector into training(2/3 of data)
# and test sets(1/3 of data)

indices = range(len(xList))
xListTest = [xList[i] for i in indices if i % 3 == 0]
xListTrain = [xList[i] for i in indices if i % 3 != 0]
labelsTest = [labels[i] for i in indices if i % 3 == 0]
labelsTrain = [labels[i] for i in indices if i % 3 != 0]

# form list of list input into numpy arrays to match input class
# for scikit-learn linear model
xTrain = numpy.array(xListTrain)
yTrain = numpy.array(labelsTrain)
xTest = numpy.array(xListTest)
yTest = numpy.array(labelsTest)

# check shapes to see what they look like
print "Shape of xTrain array", xTrain.shape
print "Shape of yTrain array", yTrain.shape

rocksVMinesModel = linear_model.LinearRegression()
rocksVMinesModel.fit(xTrain, yTrain)

# generate predictions on in-sample error
trainingPredictions = rocksVMinesModel.predict(xTrain)
print "Some values predicted by model", trainingPredictions[0:5], trainingPredictions[-6:-1]

# generate confusion matrix for prediction on training set (in-sample)
confusionMatTrain = confusionMatrix(trainingPredictions, yTrain, 0.5)

# pick threshold value and generate confusion matrix entires
tp, fn, fp, tn = confusionMatTrain

print "tp = " + str(tp) + "\tfn = " + str(fn) + "\n" + "fp = " + str(fp) + "\ttn = " + str(tn) + '\n'

# generate predictions on out-of-sample data
testPredictions = rocksVMinesModel.predict(xTest)

# gneerate confusion matrix from predictions on out-of-sample data
conMatTest = confusionMatrix(testPredictions, yTest, 0.5)
# pick threshold value and generate confusion matrix entries
tp, fn, fp, tn = conMatTest

print "tp = " + str(tp) + "\tfn = " + str(fn) + "\n" + "fp = " + str(fp) + "\ttn = " + str(tn) + '\n'

# generate ROC curve for in-sample

fpr, tpr, thresholds = roc_curve(yTrain, trainingPredictions)
roc_auc = auc(fpr, tpr)
print 'AUC for in-sample ROC curve: %f' % roc_auc

# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)

pl.plot([0, 1], [0, 1], 'k-')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])

pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('In sample ROC rocks versus mines')
pl.legend(loc="lower right")
pl.show()

# generate ROC curve for out-of-sample
fpr, tpr, thresholds = roc_curve(yTest, testPredictions)
roc_auc = auc(fpr, tpr)
print 'AUC for out-of-sample ROC curve %f' % roc_auc

# plot roc curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k-')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Out-of-sample ROC rocks versus mines')
pl.legend(loc='lower right')
pl.show()
