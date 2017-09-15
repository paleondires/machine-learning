# -*- coding: utf-8 -*-
"""
Simple Linear Regression 

Created on Fri Sep 15 08:31:12 2017

@author: pleondires
"""
# Importing the libraries 
import numpy as np  #contains mathematical tools
import matplotlib.pyplot as plt #Tools for plotting charts
import pandas as pd  #import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
# [:, :-1] - Take all of the rows, and take all of the columns -1; exclude the last column
X = dataset.iloc[:, :-1].values
# [:, 3] - Take all of the rows, take column at index 3 (Salary)
Y = dataset.iloc[:, 1]  

#Splitting the dataset into Training/Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

"""
Don't need to scale data because library will handle scaling for us
"""

#Fitting SLR to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#Regressor is the machine. Machine learns based on our provided training data
#This is one of the simplest examples of ML
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Visualizing Training Set Results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing Test Set Results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()