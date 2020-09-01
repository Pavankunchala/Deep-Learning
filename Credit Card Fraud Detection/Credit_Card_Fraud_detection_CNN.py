#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:25:34 2020

@author: pavankunchala
"""

#importing the dependencies

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('creditcard.csv')

print(data.head())
print(" ")

#checking the data if it is null
print(data.isnull().sum())

#balancing dataset
non_fraud = data[data['Class']== 0]
fraud = data[data['Class']==1]

#selecting the num of non-fraud same as fraud

non_fraud = non_fraud.sample(fraud.shape[0])


data = fraud.append(non_fraud ,ignore_index = True)
print(data.head())

X = data.drop('Class', axis =1)
y = data['Class']

#train test split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0,stratify = y)

#scaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

#reshaping the training data
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)


#building the model
ephocs =50

model = Sequential()

model.add(Conv1D(32,2,activation='relu',input_shape = X_train[0].shape))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.2))

model.add(Conv1D(64,2,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation = 'sigmoid'))

model.summary()


#compiling the model
model.compile(optimizer=Adam(lr = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, y_train,epochs=  ephocs,validation_data=(X_test,y_test), verbose = 2)














