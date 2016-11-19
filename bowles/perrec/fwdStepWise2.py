__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import numpy as np
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt
import urllib2
import time


def xattrSelect(x, idxSet):
    return [[row[i] for i in idxSet] for row in x]


def data_in(url, separator=";"):
    data = [line.strip().split(separator) for line in urllib2.urlopen(url)]
    t2 = time.time()
    return data[0], [data[i][:-1] for i in range(1, len(data))], [data[i][-1] for i in
                                                                  range(1, len(data))], t2


def dataset_creation(data, labels):
    index = range(len(data))
    dataTest = [data[i] for i in index if i % 3 == 0]
    dataTrain = [data[i] for i in index if i % 3 != 0]
    labelsTest = [labels[i] for i in index if i % 3 == 0]
    labelsTrain = [labels[i] for i in index if i % 3 != 0]
    return index, dataTest, dataTrain, labelsTest, labelsTrain


target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
t1 = time.time()
h, attributes, l, t2 = data_in(target_url)
t3 = time.time()

print "Data read took: {:.4f} s\nData return took: {:.4f} s".format(t2 - t1, t3 - t2)

t4 = time.time()
ind, attributeTestSet, attributeTrainingSet, lTestSet, lTrainingSet = dataset_creation(attributes, l)
t5 = time.time()

print "Data split took: {:.4f} s".format(t5 - t4)
