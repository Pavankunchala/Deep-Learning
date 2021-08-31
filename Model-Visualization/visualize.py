import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense , Dropout,Flatten,Input, LeakyReLU , Softmax, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import plot_model


#input
input_layer = Input(shape = (64, 64,3),name = 'input_layer')
conv_1 = Conv2D(kernel_size = (2,2),padding='same',strides=(2,2),filters = 32)(input_layer)
activation_1 = LeakyReLU(name  = 'activation_1')(conv_1)
batch_normalization_1 = BatchNormalization(name = 'batch_normalization_1')(activation_1)
pooling_1 = MaxPooling2D(pool_size = (2,2),strides=(1,1),name = 'pooling_1')(batch_normalization_1)

conv_2 = Conv2D(kernel_size = (2,2),padding ='same',strides=(2,2),filters = 64)(pooling_1)

activation_2 = LeakyReLU(name = 'activation_2')(conv_2)

batch_normalization_2 = BatchNormalization(name = 'batch_normalization_2')(activation_2)


pooling_2 = MaxPooling2D(pool_size = (2,2),strides=(1,1),name = 'pooling_2')(batch_normalization_2)

dropout = Dropout(rate = 0.5, name = 'dropout')(pooling_2)

flatten = Flatten(name = 'flatten')(dropout)

dense_1 = Dense(name = 'dense_1',units = 256)(flatten)

activation_3 = LeakyReLU(name = 'activation_3')(dense_1)

dense_2 = Dense(name = 'dense_2',units = 128)(activation_3)

activation_4 = LeakyReLU(name='activation_4')(dense_2)
dense_3 = Dense(units=3, name='dense_3')(activation_4)
output = Softmax(name='output')(dense_3)
model = Model(inputs=input_layer, outputs=output,
name='my_model')


print(model.summary())

plot_model(model,
 show_shapes=True,
 show_layer_names=True,
 to_file='my_model.jpg')
model_diagram = Image.open('my_model.jpg')










