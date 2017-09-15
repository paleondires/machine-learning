# -*- coding: utf-8 -*-
"""
Data Preprocessing - Includes all pre-processing tasks (missing data, encoding, scaling)

Created on Thu Sep 14 14:13:31 2017

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

#Encoding data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X =  onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Split the dataset into Training/Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
#Don't need to fit test set because it is already fitted to the test set.
X_test = sc_X.transform(X_test)
#Don't need to scale dependent variable because it is categorical