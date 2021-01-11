import tensorflow as tf
import numpy as np
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#let's start wity the model
model = Sequential()
model.add(Dense(units = 1, input_shape = [1]))

model.compile(optimizer = 'sgd', loss = 'mean_squared_error')



xs = np.array([1, 2, 3, 4])
ys = np.array([100, 150, 200, 250]) / 100

model.fit(xs, ys, epochs=1000)

print(model.predict([7.0]))