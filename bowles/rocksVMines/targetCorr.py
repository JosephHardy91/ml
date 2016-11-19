__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import pandas as pd
from pandas import DataFrame
from random import uniform
import matplotlib.pyplot as plot

target_url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
)

rocksVMines = pd.read_csv(target_url, header=None, prefix='V')

#assign 0 or 1 target value based on "M" or "R" labels
target=[1.0 if rocksVMines.iat[i,rocksVMines.shape[1]-1]=="M" else 0.0 for i in range(rocksVMines.shape[0])]

#plot 35th attribute
dataRow=rocksVMines.iloc[0:rocksVMines.shape[0],35]
plot.scatter(dataRow,target)

plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()

#To improve the visualization, this version dithers the points a little and makes them somewhat transparent
target=[1.0+uniform(-0.1,0.1) if rocksVMines.iat[i,rocksVMines.shape[1]-1]=="M" else 0.0+uniform(-0.1,0.1) for i in range(rocksVMines.shape[0])]

dataRow=rocksVMines.iloc[0:rocksVMines.shape[0],35]
plot.scatter(dataRow,target,alpha=0.5,s=120)

plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()