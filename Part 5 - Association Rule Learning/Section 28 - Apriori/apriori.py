# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 20:07:39 2017

@author: Nott
"""

#Apriori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset with pandas
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

#Training Apriori to dataset
#support = people buy a products 4 times * 7 day(from data) / 7500 transactions
from apyori import apriori
rules= apriori(transactions, min_support = 0.004 , min_confidence = 0.2 , min_lift = 3, min_length = 2)

#Visualising the Results
results = list(rules)
myResults = [list(x) for x in results]
