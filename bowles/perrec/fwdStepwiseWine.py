__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt
import urllib2


def columnSelect(matrix, list_of_columns):
    return np.array(matrix)[:, list_of_columns]


def read_data_from_url(url, divider=";", data_application_function=float):
    unparsed_data = urllib2.urlopen(url)
    parsed_data = [[data_application_function(val)
                    for val in line.strip().split(divider)] if li != 0 else [str(val) for val in
                                                                             line.strip().split(divider)]
                   for li, line in enumerate(unparsed_data)]

    return parsed_data[0], \
           [x[:-1] for x in parsed_data[1:]], \
           [y[-1] for y in parsed_data[1:]]


def divide_set(x, y, sectioner=3):
    r = range(len(x))
    return [[x[xi] for xi in r if xi % sectioner == 0],
            [x[xi] for xi in r if xi % sectioner != 0],
            [y[yi] for yi in r if yi % sectioner == 0],
            [y[yi] for yi in r if yi % sectioner != 0]]


def numpyify(*args):
    return [np.array(arg) for arg in args]


def fwdStepWiseRegression(xTrain1, xTest1, yTrain1, yTest1, model=sklearn.linear_model.LinearRegression()):
    attributeList = []
    indexSeq = []
    oosError = []

    indexSet = set(range(len(xTrain1[1])))
    for i in range(len(xTrain1[1])):
        errorList = []
        attTemp = []
        attTry = list(set(indexSet - set(attributeList)))
        for iTry in attTry:
            attTemp = attributeList + [iTry]
            xTrain, xTest = columnSelect(xTrain1, attTemp), columnSelect(xTest1, attTemp)
            yTrain, yTest = numpyify(yTrain1, yTest1)
            model.fit(xTrain, yTrain)

            rmsError = np.linalg.norm((yTest - model.predict(xTest)), 2) \
                       * (1.0 / np.sqrt(len(yTest)))

            errorList.append(rmsError)
            attTemp = []

        # both benefit from moving the argmin to iBest
        best_column_swathe = np.argmin(errorList)
        attributeList.append(attTry[best_column_swathe])
        oosError.append(errorList[best_column_swathe])

    return oosError, attributeList


headers, x, y = read_data_from_url(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
xTr, xTe, yTr, yTe = divide_set(x, y)
errors, best_to_worst_columns = fwdStepWiseRegression(xTr, xTe, yTr, yTe)

print "Out of sample error versus attribute set size:"
print errors
print "\n" + "Best attribute indices"
print best_to_worst_columns
namesList = [headers[i] for i in best_to_worst_columns]
print "\n" + "Best attribute names"
print namesList
