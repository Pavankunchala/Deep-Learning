import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout , Input , Flatten, concatenate , Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

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

#parameters
input_shape = (image_size, image_size, 1)
batch_size = 32
kernel_size =3
dropout = 0.4
n_filters = 32

"""
Now we are creating a Y network 
2 Inputs and 1 output 
A normal sequental network can only work with one input and one output
So thats why we are going to create a functional API model

"""


#left branch of Y network
left_inputs = Input(shape= input_shape)
x = left_inputs
filters = n_filters

"""
Create 3 layers of Conv2D , Dropout,and Maxpooling2D  
and after every loop double the filters
"""
for i in range(3):
    x = Conv2D(filters = filters,kernel_size = kernel_size,padding = 'same', activation = 'relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D()(x)
    filters *= 2
    
#lets work on the right branch of Y network
right_inputs = Input(shape = input_shape)
y = right_inputs
filters = n_filters
"""
Create 3 layers of Conv2D , Dropout,and Maxpooling2D  
and after every loop double the filters
"""
for i in range(3):
    y = Conv2D(filters = filters,kernel_size = kernel_size,padding = 'same', 
               activation = 'relu',dilation_rate =2)(y)
    y = Dropout(dropout)(y)
    y = MaxPooling2D()(y)
    filters *= 2
    
#now lets merge the 2 networsk
y = concatenate([x,y])

#now lets start with thr dense layers
y = Flatten()(y)
y = Dropout(dropout)(y)
outputs = Dense(num_labels , activation = 'softmax')(y)


#now lets build the model
model = Model(inputs = [left_inputs, right_inputs], outputs = outputs)

plot_model(model, to_file='cnn-Y-network.png', show_shapes=True)
# verify the model using layer text description
model.summary()

model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

model.fit([x_train, x_train],
             y_train,
             validation_data=([x_test, x_test], y_test),
             epochs=15,
             batch_size=batch_size)


score = model.evaluate([x_test, x_test],
                          y_test,
                          batch_size=batch_size,
                          verbose=2)

print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))









    
