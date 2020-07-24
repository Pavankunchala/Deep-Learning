import tensorflow as tf

import numpy as np

from tensorflow import keras

#defining a model

"here we are trying to find a relation between X and Y and you can see  y = 2x -1 (more probable)"



model = tf.keras.Sequential([keras.layers.Dense(units = 1,input_shape =[1])])


model.compile(optimizer='sgd', loss='mean_squared_error')


xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


#training A neural network

model.fit(xs, ys, epochs=500)

#prediction of the model

print("The prediction is ",model.predict([10.0]))