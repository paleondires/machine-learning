# -*- coding: utf-8 -*-
"""
K-Means 
Created on Fri Sep 22 13:55:45 2017

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

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('# Clusters')
plt.ylabel('WCSS')
plt.show()
    
#Assign kmeans using the ideal # of clusters based on plot above.
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, color = 'red', label = 'Careful', edgecolors = 'black')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, color = 'blue', label = 'Standard', edgecolors = 'black')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, color = 'green', label = 'Target', edgecolors = 'black')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, color = 'cyan', label = 'Careless', edgecolors = 'black')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, color = 'magenta', label = 'Sensible', edgecolors = 'black')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids', edgecolors = 'black')
plt.title('Client Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()