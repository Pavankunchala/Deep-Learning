#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 19:04:38 2020

@author: pavankunchala
"""



import tensorflow as tf


#importing the dataset
mnist = tf.keras.datasets.fashion_mnist


(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=200)

#callbacks

class myCallback(tf.keras.callbacks.Callback):
    
    def on_epoch_end(self,epoch,logs = {}):
        
        
        
        if(logs.get('accuracy')>0.6):
            
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True

#just for the sake visualization

plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])


#now  normalizing




training_images = training_images/255
test_images = test_images/255

#a sequential model with 3 layers ,flatten,dense,and other dense layer

model = tf.keras.Sequential([tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(128,activation = tf.nn.relu),
                             tf.keras.layers.Dense(10,activation = tf.nn.softmax)
                             ])


model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

callbacks = myCallback()


model.fit(training_images, training_labels, epochs=10,callbacks = [callbacks])

model.evaluate(test_images, test_labels)


classifications = model.predict(test_images)

print(classifications[0])


print(test_labels[0])

