# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 10:21:03 2017

@author: Nott
"""

#Artificial  Neural Network

#Installing Theano
#pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

#Installing Tenserflow
#pip install --upgrade tensorflow

#Install Keras
#pip install --upgrade keras

#Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Create Dataset
Nclass = 500
D = 2 # dimensionality of input


X1 = np.random.randn(Nclass, D) + np.array([0, -2])
X2 = np.random.randn(Nclass, D) + np.array([2, 2])
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
X = np.vstack([X1, X2, X3]).astype(np.float32)

y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
                
# Encoding categorical data
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]"""


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#OneHotEncode y_train
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part 2 - Let's make ANN

#Import Keras Libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising ANN
classifier = Sequential()

#Adding the Input layer and First hidden layers
classifier.add(Dense(output_dim=3,init='uniform',activation='relu',input_dim=2))

#Adding Output Layer   if dependencies param > 2  change output_dim=x and activation=softmax
classifier.add(Dense(output_dim=3,init='uniform',activation='softmax'))

#Part 3 - Make Prediction and Evaluating the model if dependencies param > 2  change loss='categorical_crossentropy'
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10 ,nb_epoch=25)

# Predicting the Test set results
classes = classifier.predict_classes(X_test, batch_size=10)
proba = classifier.predict_proba(X_test, batch_size=10)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
y_test= y_test.astype(int)
cm = confusion_matrix(y_test, classes)









































