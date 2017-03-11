# -*- coding: utf-8 -*-
"""
Spyder Editor


This is a temporary script file.
"""


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


#Create Data
x_asc=np.array(np.random.randint(50,1000, size=(50000,9)))
x_asc.sort(axis=1)
x_asc_noise = np.array(np.random.randint(400,500, size=(1,50000)))
x_asc = np.concatenate((x_asc, x_asc_noise.T), axis=1)
y_asc=np.array(np.random.randint(1,2, size=(1,50000)))
x_y_asc = np.concatenate((x_asc, y_asc.T), axis=1)

x_desc=np.array(np.random.randint(50,1000, size=(50000,10)))
#x_desc[:,::-1].sort(axis=1)
y_desc=np.array(np.random.randint(0,1, size=(1,50000)))
x_y_desc = np.concatenate((x_desc, y_desc.T), axis=1)
dataset = np.concatenate((x_y_asc, x_y_desc))

# Importing the dataset
X = dataset[:, :-1]
y = dataset[:, 10]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

#Reshape for LSTM
X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# create the model
model = Sequential()
model.add(GRU(200, input_shape=X_train.shape[1:]))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, batch_size=10, nb_epoch=3, validation_data=(X_test, y_test))

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))