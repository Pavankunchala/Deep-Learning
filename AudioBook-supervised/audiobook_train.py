#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:15:51 2020

@author: pavankunchala
"""

import numpy as np
from sklearn import preprocessing
import tensorflow as tf


raw_csv_data  = np.loadtxt('Audiobooks_data.csv',delimiter=',')


unscaled_inputs_all = raw_csv_data[:,1:-1]
targets_all = raw_csv_data[:,-1]

#balancing the dataset

num_one_targets = int(np.sum(targets_all))

zero_targets_counter = 0
indices_to_remove = []


for i in range(targets_all.shape[0]):
    if targets_all[i] ==0:
        zero_targets_counter +=1
        if zero_targets_counter> num_one_targets:
            indices_to_remove.append(i)
            
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all,indices_to_remove,axis =0)
targets_equal_priors = np.delete(targets_all,indices_to_remove,axis = 0)

#Standardinzing the inputs
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

#shuffle the data
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]

#spltiing into train and test
samples_count = shuffled_inputs.shape[0]


train_samples_count = int(0.8*samples_count)
validation_samples_count  = int(0.1*samples_count)
test_samples_count  = samples_count - train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count: train_samples_count + validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+ validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count + validation_samples_count:]
test_targets = shuffled_targets[train_samples_count + validation_samples_count:]

#saving as npz
np.savez('Audiobooks-data-train',inputs = train_inputs,targets = train_targets)
np.savez('Audiobooks-data-validation',inputs = validation_inputs,targets = validation_targets)
np.savez('Audiobooks-data-test',inputs = test_inputs,targets = test_targets)


#loading the preprocessed data
npz= np.load("Audiobooks-data-train.npz")
 
train_inputs = npz['inputs'].astype(np.float)
train_targets = npz['targets'].astype(np.int)

npz= np.load("Audiobooks-data-validation.npz")

validation_inputs, validation_targets =  npz['inputs'].astype(np.float) , npz['targets'].astype(np.int)


npz= np.load("Audiobooks-data-test.npz")

test_inputs, test_targets =  npz['inputs'].astype(np.float) , npz['targets'].astype(np.int)



#Creating the model

input_size = 10

output_size =2

hidden_layers_size= 50


model = tf.keras.Sequential([
  tf.keras.layers.Dense(hidden_layers_size,activation = 'relu'),
  tf.keras.layers.Dense(hidden_layers_size,activation = 'relu'),
  tf.keras.layers.Dense(output_size,activation = 'softmax')
    
    ])

#compiling the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 100

max_ephocs = 100

early_stopping = tf.keras.callbacks.EarlyStopping(patience=3)

model.fit(train_inputs,
          train_targets,
          batch_size=batch_size,
          epochs= max_ephocs,
          callbacks=[early_stopping],
          validation_data=(validation_inputs,validation_targets),
          verbose =2
          )


#testing the funciton

test_loss ,test_accuarcy = model.evaluate(test_inputs,test_targets)


print('Test loss :', test_loss)
print("Test accuarcy",test_accuarcy)

















            
        
    
    