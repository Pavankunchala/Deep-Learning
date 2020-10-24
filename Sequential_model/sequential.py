import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(Flatten(input_shape = (28,28)))
model.add(Dense(16,activation= 'relu'))
model.add(Dense(16,activation= 'relu'))
model.add(Dense(10,activation= 'softmax'))

# Print the model summary
model.summary()

# Build the Sequential convolutional neural network model
model = Sequential()

model.add(Conv2D(16,(3,3),activation='relu',padding = 'SAME', input_shape = (28,28,3))) # a color image of (28,28)
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))

# Print the model summary
model.summary()
