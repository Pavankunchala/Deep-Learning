import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

# Create features
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Visualize it
plt.scatter(X, y)

#plt.show()

#let's create tensors for demo input and output shape
house_info = tf.constant(['Bedroom','Bathroom','Garage'])
house_price = tf.constant([840000])

print('House info',house_info)

print('House price',house_price)

#let's create a small model 

#set random seed
tf.random.set_seed(42)

#creating a model for sequential api
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(1))

#compile the model
model.compile(loss = tf.keras.losses.mae,
              optimizer = tf.keras.optimizers.SGD(),
              metrics = ['mae'])

model.fit(X,y, epochs= 5,verbose = 2)

