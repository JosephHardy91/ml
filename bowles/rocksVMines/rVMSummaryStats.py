__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os, urllib2
import matplotlib.pyplot as plt
import numpy as np
# read data from uci data repository
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
    xList.append(row)
nrow = len(xList)
ncol = len(xList[1])

type = [0] * 3
colCounts = []

# generate summary statistics for column 3 (e.g.)
col = 3
colData = []
for row in xList:
    colData.append(float(row[col]))

colArray = np.array(colData)
colMean = np.mean(colArray)
colsd = np.std(colArray)
sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' +
                 "Standard Deviation = " + '\t' + str(colsd) + "\n")

# calculate quantile boundaries
ntiles = 4

percentBdry = []

for i in range(ntiles + 1):
    percentBdry.append(np.percentile(colArray, i * (100) / ntiles))

sys.stdout.write("\nBoundaries for {ntiles} Equal Percentiles \n".format(ntiles=ntiles))
print(percentBdry)
sys.stdout.write(" \n")

# do again with 10
ntiles = 20

percentBdry = []

for i in range(ntiles + 1):
    percentBdry.append(np.percentile(colArray, i * (100) / ntiles))

sys.stdout.write("\nBoundaries for {ntiles} Equal Percentiles \n".format(ntiles=ntiles))
print(percentBdry)
sys.stdout.write(" \n")
# plt.scatter(range(ntiles+1), percentBdry)
# plt.plot([0]+range(ntiles+1)+[ntiles+.0000001],[0]+percentBdry+[0],color='red')
# plt.show()
# Last column contains categorical variables

col = 60
colData = []
for row in xList:
    colData.append(row[col])

unique = set(colData)
sys.stdout.write("Unique Label Values \n")
print(unique)

# count up the number of elements having each value
catDict = dict(zip(list(unique), range(len(unique))))
catCount = [0] * len(unique)

for element in colData:
    catCount[catDict[element]] += 1

sys.stdout.write("\nCounts for Each Value of Categorical Label \n")
print(list(unique))
print(catCount)
