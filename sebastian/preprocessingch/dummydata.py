__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import pandas as pd
from io import StringIO

csv_data = """A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,"""

csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))

if __name__ == "__main__":
    print df
    # display number of missing values per column
    print df.isnull().sum()
    # to drop rows with null values
    df.dropna()
    # to drop columns with null values
    df.dropna(axis=1)
    # drop rows/columns where all columns/rows are NaN
    df.dropna(how='all')

    # drop rows that have not at least 4 non-NaN values
    df.dropna(thresh=4)

    #only drop rows where NaN appears in specific columns
    df.dropna(subset=['C'])

    # df.values keeps the numpy value arrays of the dataframe
    print df.values