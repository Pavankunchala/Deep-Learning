from re import M
import tensorflow as tf
from tensorflow.keras.layers import Dense , Conv2D, Flatten, Input , Reshape , Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

import numpy as np
import  matplotlib.pyplot as plt


(x_train, _), (x_test, _) = mnist.load_data()

# reshape to (28, 28, 1) and normalize input images
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#network parameters
input_shape = (image_size, image_size, 1)
batch_size = 32
kernel_size = 3
latent_dim = 16

# encoder/decoder number of CNN layers and filters per layer
layer_filters = [32, 64]

"""
Let's Start building our Autoencode model
but before that lets create the encoder model 

"""
inputs = Input( shape = input_shape , name = 'Encoder_input')

x = inputs

# stack of Conv2D(32)-Conv2D(64)

for filters in layer_filters:
    x = Conv2D(filters = filters, kernel_size =kernel_size,
               activation = 'relu',strides = 2 , padding = 'same')(x)
    
"""
we need the shape info for building decoder model

"""
shape = K.int_shape(x)

#lets create the latent vector
x = Flatten()(x)
latent = Dense(latent_dim,name = 'latent_vector')(x)

#instantiate the Encoder ModelC
encoder = Model(inputs,latent,name = 'encoder')

encoder.summary()
plot_model(encoder,
              to_file='encoder_MNIST.png',
              show_shapes=True)

#lets build the decoder model
latent_inputs = Input(shape = (latent_dim,), name = 'decoder_input')

x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)

# from vector to suitable shape for transposed conv
x = Reshape((shape[1], shape[2], shape[3]))(x)

# stack of Conv2DTranspose(64)-Conv2DTranspose(32)
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters = filters, kernel_size = kernel_size,
                activation='relu', strides = 2, padding = 'same')(x)
    
#reconstruct the input 

outputs = Conv2DTranspose(filters = 1 ,kernel_size = kernel_size,activation='sigmoid',
                          padding = 'same', name = 'decoder_output')(x)

#instantiate the decoder model
decoder = Model(latent_inputs, outputs, name = 'decoder')

decoder.summary()

plot_model(decoder, to_file='decoder_MNIST.png', show_shapes=True)

#autoencoder = Encoder + decoder

autoencoder = Model(inputs, decoder(encoder(inputs)),name = 'autoencoder')

autoencoder.summary()

plot_model(autoencoder,
           to_file='autoencoder_MNIST.png',
           show_shapes=True)

autoencoder.compile(loss='mse', optimizer='adam')

autoencoder.fit(x_train,
                x_train,
                validation_data=(x_test, x_test),
                epochs=1,
                batch_size=batch_size)

x_decoded = autoencoder.predict(x_test)

imgs = np.concatenate([x_test[:8], x_decoded[:8]])
imgs = imgs.reshape((4, 4, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Input: 1st 2 rows, Decoded: last 2 rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('input_and_decoded.png')
plt.show()

