import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout
from tensorflow.keras.utils import to_categorical , plot_model
from tensorflow.keras.datasets import mnist
import pydot

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# compute the number of labels
num_labels = len(np.unique(y_train))


#conveting to one hot encoder 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
input_size = image_size * image_size

# resize and normalize
x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255

#parameters
batch_size =128
hidden_units =256
dropout= 0.45

model = Sequential()
#layer 1
model.add(Dense(units = hidden_units , activation = 'relu', input_dim = input_size))
model.add(Dropout(dropout))
#layer 2
model.add(Dense(units = hidden_units , activation = 'relu'))
model.add(Dropout(dropout))
#layer 3
model.add(Dense(units = num_labels, activation = 'softmax'))

print(model.summary())

plot_model(model, to_file='MLP-mnist.png', show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

_, acc = model.evaluate(x_test, y_test,  batch_size=batch_size,verbose = 2)

print("\nTest accuracy: %.1f%%" % (100.0 * acc))


