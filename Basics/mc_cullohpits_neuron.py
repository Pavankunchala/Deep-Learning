#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:25:28 2020

@author: pavankunchala
"""
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam, SGD

#Input data
X_train = [.1,.2,.3, .4]
Y_train= [.05,.1, .15,.2]

print("X_train ", X_train)
print("Y_train ", Y_train)


#Model architecture
modelSimple = Sequential()
modelSimple.add(Dense(1, init='uniform', input_shape=(1,)))

#Compile and fit model
LEARNING_RATE =0.05
modelSimple.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mse')
modelSimple.fit(X_train, Y_train, batch_size=1, nb_epoch=20, verbose=1, validation_split=0.2) 

#Print weights
print("")
print("Weights: \n")
print(modelSimple.get_weights())

#Print prediction
print("")
print("Prediction: \n")
print(modelSimple.predict(X_train, batch_size=1))

