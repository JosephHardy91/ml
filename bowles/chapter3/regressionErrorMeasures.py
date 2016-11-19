__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os

target = [1.5, 2.1, 3.3, -4.7, -2.3, 0.75]
prediction = [0.5, 1.5, 2.1, -2.2, 0.1, -0.5]

error = []
for i in range(len(target)):
    error.append(target[i] - prediction[i])

# print the errors
print "Errors"
print error


# calculate teh squared errors and absolute value of errors
squaredError = []
absError = []
for val in error:
    squaredError.append(val ** 2)
    absError.append(abs(val))

# print squared errors and absolute value of errors
print "Squared Error"
print squaredError

print "Absolute Value of Error"
print absError

# calculate and print mean squared error MSE
print "MSE = ", sum(squaredError) / len(squaredError)

from math import sqrt
# calculate and print square root of MSE (RMSE)
print "RMSE = ", sqrt(sum(squaredError) / len(squaredError))

# calculate and print mean absolute error MAE
print "MAE = ", sum(absError) / len(absError)

# compare MSE to target variance
targetDeviation = []
targetMean = sum(target) / len(target)
for val in target:
    targetDeviation.append((val - targetMean) ** 2)

# print the target variance
print "Target Variance = ", sum(targetDeviation) / len(targetDeviation)

# print the target standard deviation (square root of variance)
print "Target Standard Deviation = ", sqrt(sum(targetDeviation) / len(targetDeviation))
