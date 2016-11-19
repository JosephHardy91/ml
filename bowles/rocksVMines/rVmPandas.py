__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot

target_url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
)

rocksVMines = pd.read_csv(target_url, header=None, prefix='V')

#print head and tail of data frame
print(rocksVMines.head())
print(rocksVMines.tail())

#print summary of data frame
summary=rocksVMines.describe()
print(summary)


