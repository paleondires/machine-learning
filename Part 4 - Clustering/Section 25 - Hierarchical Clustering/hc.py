# -*- coding: utf-8 -*-
"""
Hierarchical Clustering
Created on Fri Sep 22 14:58:47 2017

@author: pleondires
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the mall dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
#No Y because this is an unsupervised learning example. We don't know what we're looking for


#Using dendrogram to find optimal # of clusters
import scipy.cluster.hierarchy as sch
#Ward method minimizes variance in each cluster. 
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel ('Euclidean Distances')
plt.show()

#Determined the optimal # of clusters is 5 based on dendrogram above
#Fitting HC to the dataset
from sklearn.cluster import AgglomerativeClustering
# Use same linkage, and use euclidean distance to compute said linkage
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, color = 'red', label = 'Careful', edgecolors = 'black')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, color = 'blue', label = 'Standard', edgecolors = 'black')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, color = 'green', label = 'Target', edgecolors = 'black')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, color = 'cyan', label = 'Careless', edgecolors = 'black')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, color = 'magenta', label = 'Sensible', edgecolors = 'black')
plt.title('Client Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()