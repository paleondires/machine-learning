# -*- coding: utf-8 -*-
"""
Example for handling Missing data
Created on Fri Sep 15 10:58:36 2017

@author: pleondires
"""

# Importing the libraries 
import numpy as np  #contains mathematical tools
import matplotlib.pyplot as plt #Tools for plotting charts
import pandas as pd  #import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# [:, :-1] - Take all of the rows, and take all of the columns -1; exclude the last column
X = dataset.iloc[:, :-1].values
# [:, 3] - Take all of the rows, take column at index 3 (Purchased)
Y = dataset.iloc[:, 3]  

#Handling missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# Take columns 1,2 (1 is inclusive bound, 3 is exclusive bound)
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])