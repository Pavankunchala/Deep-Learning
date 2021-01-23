import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import  Dense , Conv2D, MaxPooling2D , Dropout, Input,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# compute the number of labels
num_labels = len(np.unique(y_train))


#conveting to one hot encoder 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# reshape and normalize input images
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#  parameters
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
filters = 64
dropout = 0.3

#functional API for building CNN
inputs = Input(shape = input_shape)

"""
Y is the output tensor where as X is the input tensor
in let's say a function 
Y = Conv2D(32)(X)
"""
y = Conv2D(filters = filters , kernel_size =kernel_size , activation = 'relu')(inputs)
y = MaxPooling2D()(y)

y = Conv2D(filters = filters , kernel_size =kernel_size , activation = 'relu')(y)
y = MaxPooling2D()(y)

y = Conv2D(filters = filters , kernel_size =kernel_size , activation = 'relu')(y)

#flatten the image before dense layers
y = Flatten()(y)
#reguarlize it for reducing the parameters
y = Dropout(dropout)(y)

outputs = Dense(num_labels, activation = 'softmax')(y)

#building the model by supplying inputs and outputs
model = Model(inputs= inputs, outputs= outputs)

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(x_train,
             y_train,
             validation_data=(x_test, y_test),
             epochs=20,
             batch_size=batch_size)

score = model.evaluate(x_test,
                          y_test,
                          batch_size=batch_size,
                          verbose=2)

print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))




