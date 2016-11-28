__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
from dummydata import *
from sklearn.preprocessing import Imputer

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)  # axis=0 columns, axis=1 rows
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print imputed_data
