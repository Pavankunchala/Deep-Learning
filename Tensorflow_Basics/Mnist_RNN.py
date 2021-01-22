import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.utils import to_categorical , plot_model
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# compute the number of labels
num_labels = len(np.unique(y_train))


#conveting to one hot encoder 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size])
x_test = np.reshape(x_test,[-1, image_size, image_size])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


input_shape = (image_size, image_size)
batch_size = 128
units = 256
dropout = 0.2


model = Sequential()
model.add(SimpleRNN(units=units, dropout=dropout, input_shape=input_shape))
model.add(Dense(units = num_labels, activation = 'softmax'))
model.summary()
plot_model(model, to_file='rnn-mnist.png', show_shapes=True)


model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=batch_size)
_, acc = model.evaluate(x_test,y_test, batch_size=batch_size, verbose=1)

print("\nTest accuracy: %.1f%%" % (100.0 * acc))


