# -*- coding: utf-8 -*-
"""
SVM

Finding best decision boundary

Created on Wed Sep 20 17:41:06 2017

@author: pleondires
"""
# Importing the libraries 
import numpy as np  #contains mathematical tools
import matplotlib.pyplot as plt #Tools for plotting charts
import pandas as pd  #import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
#Age and Salary
X = dataset.iloc[:, [2,3]].values
#Purchased?
Y = dataset.iloc[:, 4]


#Split the dataset into Training/Test Set  
#80/20 Training/Test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting classifier
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix   - 65, 24 Correct predictions  (89) and 8,3 incorrect (11)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Visualization
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j, 1], c =ListedColormap(('red', 'green'))(i), label = j, edgecolors = 'black')
plt.title('SVM (Training)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j, 1], c =ListedColormap(('red', 'green'))(i), label = j, edgecolors = 'black')
plt.title('SVM (Test)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()