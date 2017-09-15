# -*- coding: utf-8 -*-
"""
Example for encoding categorical data

Created on Fri Sep 15 10:59:35 2017

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

#Encoding data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X =  onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)