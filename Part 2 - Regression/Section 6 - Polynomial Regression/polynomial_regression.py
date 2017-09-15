# -*- coding: utf-8 -*-
"""
Polynomial Regression

Created on Fri Sep 15 11:12:56 2017

@author: pleondires
"""
# Importing the libraries 
import numpy as np  #contains mathematical tools
import matplotlib.pyplot as plt #Tools for plotting charts
import pandas as pd  #import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#Create X as a matrix (2 is excluded because its the upper bound)
#Best practice to create X as matrix and Y as a vector
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fitting Linear regression to the dataset as a reference
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() 
lin_reg.fit(X, y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#Visualizing the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title('Level Salaries')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title('Level Salaries')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

#Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))
