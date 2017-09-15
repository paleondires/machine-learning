# -*- coding: utf-8 -*-
"""
Template for data preprocessing

Created on Fri Sep 15 10:53:22 2017

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


#Split the dataset into Training/Test Set  
#80/20 Training/Test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)