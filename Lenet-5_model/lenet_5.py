import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Conv2D,MaxPooling2D, Flatten,Dense

num_classes = None

#creating the Lenet-5 model
model = Sequential()
# 1 st Block
model.add(Conv2D(6,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# 2nd Block
model.add(Conv2D(6,kernel_size=(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#Dense layers
model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))

