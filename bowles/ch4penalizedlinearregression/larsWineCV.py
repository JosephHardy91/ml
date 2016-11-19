__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import urllib2
import numpy as np
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt
from tqdm import tqdm


# LEAST ANGLE REGRESSION
def lars(data):
    xList = []
    labels = []
    names = []
    firstLine = True
    for line in data:
        if firstLine:
            names = line.strip().split(";")
            firstLine = False
        else:
            # split on semi-colon
            row = line.strip().split(";")
            # put labels in separate array
            labels.append(float(row.pop()))
            floatRow = map(float, row)
            xList.append(floatRow)

    # Normalize columns in x and labels
    nrows = len(xList)
    ncols = len(xList[0])
    # calculate means and variances
    xMeans = []
    xSD = []
    for i in range(ncols):
        col = [xList[j][i] for j in range(nrows)]
        mean = sum(col) / nrows
        xMeans.append(mean)

        colDiff = [(xList[j][i] - mean) for j in range(nrows)]
        sumSq = sum([colDiff[i] ** 2 for i in range(nrows)])
        stdDev = sqrt(sumSq / nrows)
        xSD.append(stdDev)

    # use calculated mean and standard deviation to normalize xList
    xNormalized = []
    for i in range(nrows):
        rowNormalized = [(xList[i][j] - xMeans[j]) / xSD[j] for j in range(ncols)]
        xNormalized.append(rowNormalized)

    # normalize labels
    meanLabel = sum(labels) / nrows
    sdLabel = sqrt(sum([(labels[i] - meanLabel) ** 2 for i in range(nrows)]) / nrows)
    labelNormalized = [(labels[i] - meanLabel) / sdLabel for i in range(nrows)]

    # build cross-validation loop to determine best coefficient values

    # number of cross-validation folds
    nxval = 10

    # number of steps to take
    nSteps = 350
    stepSize = 0.004

    # initialize list for storing errors
    errors = []
    for i in range(nSteps):
        b = []
        errors.append(b)

    for ixval in tqdm(range(nxval)):
        # define test an dtrainin index sets
        idxTest = [a for a in range(nrows) if a % nxval == ixval * nxval]
        idxTrain = [a for a in range(nrows) if a % nxval != ixval * nxval]

        # define test and training attribute and label sets
        xTrain = [xNormalized[r] for r in idxTrain]
        xTest = [xNormalized[r] for r in idxTest]
        labelTrain = [labelNormalized[r] for r in idxTrain]
        labelTest = [labelNormalized[r] for r in idxTest]

        # train LARS regression on Training Data
        nrowsTrain = len(idxTrain)
        nrowsTest = len(idxTest)

        # initialize a vector of coefficients beta
        betas = [0.0] * ncols
        # initialize matrix of betas at each step
        betaMat = []
        betaMat.append(list(betas))

        for iStep in range(nSteps):
            # calculate residuals
            residuals = [0.0] * nrows
            for j in range(nrowsTrain):
                # print len(xNormalized), len(betas)
                labelsHat = sum([xTrain[j][k] * betas[k] for k in range(ncols)])
                # real minus the model ... how far away from prediction is reality
                residuals[j] = labelTrain[j] - labelsHat

            # calculate correlation between attribute columns from normalized wine and residual
            corr = [0.0] * ncols

            for j in range(ncols):
                corr[j] = sum([xTrain[k][j] * residuals[k] for k in range(nrowsTrain)]) / nrowsTrain

            iStar = 0
            corrStar = corr[0]

            for j in range(1, ncols):
                if abs(corrStar) < abs(corr[j]):
                    iStar = j
                    corrStar = corr[j]

            betas[iStar] += stepSize * corrStar / abs(corrStar)
            betaMat.append(list(betas))

            # use beta just calculated to predict and accumulate
            # out of sample error (e.g. data not being used for the calculation of beta)
            for j in range(nrowsTest):
                labelsHat = sum([xTest[j][k]  * betas[k] for k in range(ncols)])
                err = labelTest[j] - labelsHat
                errors[iStep].append(err)

    # for step in range(nSteps):
    #     ys = [errors[step][i] for i in range(len(errors[step]))]
    #     plt.plot(range(len(ys)),ys)
    cvCurve = []
    # print errors[0]
    for errVect in errors:

        mse = sum([x**2 for x in errVect])/len(errVect)
        cvCurve.append(mse)

    minMse = min(cvCurve)
    minPt = [i for i in range(len(cvCurve)) if cvCurve[i] == minMse][0]
    print "Minimum Mean Square Error",minMse
    print "Index of Minimum Mean Square Error",minPt

    xaxis = range(len(cvCurve))
    plt.plot(xaxis,cvCurve)

    plt.xlabel("Steps taken")
    plt.ylabel("Mean Square Error")

    # # get correlation
    # sstotal = sum([lN ** 2 for lN in labelNormalized])
    # ssresidual = sum(residuals)
    # xN0 = [xNormalized[i][0] for i in range(nrows)]
    # print "Sum of Squared Labels: {0}\nResiduals:{1}\nCoefficient of Determination:{2}".format(sstotal, ssresidual,
    #                                                                                            1.0 - (
    #                                                                                                ssresidual / sstotal))
    # for i in range(ncols):
    #     # plot range of beta values for each attribute
    #     coefCurve = [betaMat[k][i] for k in range(nSteps)]
    #     # print coefCurve
    #     xaxis = range(nSteps)
    #     plt.plot(xaxis, coefCurve)
    # plt.xlabel("Steps taken")
    # plt.ylabel("Coefficient Values")
    # plt.show()

    return plt


target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/" \
             "wine-quality/winequality-red.csv"
data = urllib2.urlopen(target_url)

plot = lars(data)

#print b
plot.show()
