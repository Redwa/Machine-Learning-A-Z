# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 21:37:17 2017

@author: Nott
"""

#Hierarchical Clustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
                
#Using Dendrogram to find Optimal Cluster numbers
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.plot()

#Fiting Hierarchical Clustering to the Mall dataset
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=5, affinity='euclidean',linkage='ward')
y_hc = ac.fit_predict(X)

#Visualising the Cluster
plt.scatter(X[y_hc==0,0],X[y_hc==0,1], s = 10,c = 'red', label = 'Careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1], s = 10,c = 'blue', label = 'Standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1], s = 10,c = 'green', label = 'Target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1], s = 10,c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1], s = 10,c = 'magenta', label = 'Sensible')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1 - 100)')
plt.legend()
plt.show()





