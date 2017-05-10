# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 22:40:27 2017

@author: Nott
"""

#CNN
#Installing Theano
#pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

#Installing Tenserflow
#pip install --upgrade tensorflow

#Install Keras
#pip install --upgrade keras

#Part 1 - Building the Convolutional Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

#Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#Adding a second Convolution Layer
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) #'softmax' if multiple category output

#Compiling the CNN
classifier.compile(optimizer='adam',loss = 'binary_crossentropy', metrics=['accuracy']) #'categorical_crossentropy' if multiple category output

#Part 2 - Fitting the CNN to Images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('dataset/training_set',
                                              target_size=(64, 64),
                                              batch_size=32,
                                              class_mode='binary') #'categorical' if multiple category output

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary') #'categorical' if multiple category output

classifier.fit_generator(train_set,
                         samples_per_epoch=8000,
                         nb_epoch=25,
                         validation_data=test_set,
                         nb_val_samples=800)


#Makeing new Predictions
import numpy as np
from keras.preprocessing import image as image_utils
test_image = image_utils.load_img('dataset/test_image/cat.4003.jpg', target_size=(64, 64))
#test_image = image_utils.load_img('dataset/training_set/dogs/dog.8.jpg', target_size=(64, 64))
#test_image = image_utils.load_img('dataset/training_set/cats/cat.10.jpg', target_size=(64, 64))
test_image = image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'














#