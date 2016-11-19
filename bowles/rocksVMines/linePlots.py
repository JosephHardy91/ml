__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import pandas as pd
import matplotlib.pyplot as plot
target_url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
)

#read rocks versus mines data into pandas df
rocksVMines=pd.read_csv(target_url,header=None,prefix="V")

for i in range(208):
    #assign color based on "M" or "R" labels
    if rocksVMines.iat[i,rocksVMines.shape[1]-1]=='M':
        pcolor='red'
    else:
        pcolor='blue'

    #plot rows of data as if they were series data
    dataRow=rocksVMines.iloc[i,0:rocksVMines.shape[1]-1]
    dataRow.plot(color=pcolor)

plot.xlabel("Attribute Index")
plot.ylabel(("Attribute Values"))
plot.show()
