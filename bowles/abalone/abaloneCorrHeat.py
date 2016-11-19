__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import pandas as pd
from pandas import DataFrame
#from pylab import *
from math import exp
import matplotlib.pyplot as plot
import numpy as np
import seaborn

target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")

abalone = pd.read_csv(target_url, header=None, prefix="V")
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Wt', 'Shucked Wt', 'Viscera Wt',
                   'Shell Wt', 'Rings']

corMat=DataFrame(abalone.iloc[:,1:9].corr())
#print correlation matrix
print(corMat)

mask = np.zeros_like(corMat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#visualize correlations using heatmap
# plot.pcolor(corMat,mask=mask)
# plot.show()


#Set up the matplotlib figure
f, ax = plot.subplots(figsize=(11, 9))

#Generate a custom diverging colormap
cmap = seaborn.diverging_palette(32,133,l=67,as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
seaborn.heatmap(corMat, mask=mask, cmap=cmap, #vmax=.3,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

seaborn.plt.show()