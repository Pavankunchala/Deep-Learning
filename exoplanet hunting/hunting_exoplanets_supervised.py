#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:58:06 2020

@author: pavankunchala
"""

#importing the stuff

import pandas as pd
import numpy as np
import tensorflow as tf

train = pd.read_csv('exoTrain.csv')

test = pd.read_csv('exoTest.csv')

#importing the data
#we have downloaded the data from kaggle exoplanet hunting



print('Training data\n',train.head())
print('Test data\n',test.head())

x_train = train.drop('LABEL', axis=1)
y_train = train.LABEL-1 #subtract one because of TGBT 
x_test = test.drop('LABEL', axis=1)
y_test = test.LABEL-1


numeric_column_headers = x_train.columns.values.tolist()

bc_fn = tf.feature_column.bucketized_column
nc_fn = tf.feature_column.numeric_column
bucketized_features = [bc_fn(source_column=nc_fn(key=column),
boundaries=[x_train[column].mean()]) for column in numeric_column_headers]

all_features = bucketized_features


batch_size = 32
pi_fn = tf.estimator.inputs.pandas_input_fn
train_input_fn = pi_fn(x = x_train,y = y_train,batch_size = batch_size, shuffle = True, num_epochs = None)


eval_input_fn = pi_fn(x = x_test, y = y_test,
batch_size = batch_size, shuffle = False, num_epochs = 1)


n_trees = 100
n_steps = 100
m_fn = tf.estimator.BoostedTreesClassifier
model = m_fn(feature_columns=all_features,n_trees = n_trees, n_batches_per_layer = batch_size, model_dir='./tfbtmodel')
 



model.train(input_fn=train_input_fn, steps=n_steps)

results = model.evaluate(input_fn=eval_input_fn)


for key,value in sorted(results.items()): print('{}: {}'.format(key, value))















