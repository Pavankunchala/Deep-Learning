
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, Input , Flatten
from tensorflow.keras.layers import Reshape , Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as  K
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(1337)

# load the dataset
(x_train, _), (x_test, _) = mnist.load_data()

# reshape to (28, 28, 1) and normalize input images
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


"""
Now let's add noise to the images to corrupt them and after that we can use Denoising autoencoder 
for generating new images without noise

"""
noise = np.random.normal(loc = 0.5 , scale = 0.5 , size = x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc = 0.5 , scale = 0.5 , size = x_test.shape)
x_test_noisy = x_test + noise

"""
We are not done yet sometimes adding the noise might exceed  the pixel values >1.0 or <0.0
So let's clip them to 1 and 0 respectively
"""

x_train_noisy = np.clip(x_train_noisy,0.,1.)
x_test_noisy = np.clip(x_test_noisy,0.,1.)

# network parameters
input_shape = (image_size, image_size, 1)
batch_size = 32
kernel_size = 3
latent_dim = 16
inputs = Input( shape = input_shape , name = 'Encoder_input')
# encoder/decoder number of CNN layers and filters per layer
layer_filters = [32, 64]

"""
Let's Start building the autoEncoder model

Start wit the encoder models

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

autoencoder = Model(inputs, decoder(encoder(inputs)),name = 'autoencoder')

autoencoder.summary()

# MSE as loss function and Adam  as optimizer

autoencoder.compile(loss='mse', optimizer='adam')

#train the autoencoder 


autoencoder.fit(x_train_noisy,
                x_train,
                validation_data=(x_test_noisy, x_test),
                epochs=10,
                batch_size=batch_size)


# predict the autoencoder output from corrupted test images
x_decoded = autoencoder.predict(x_test_noisy)

# 3 sets of images with 9 MNIST digits
# 1st rows - original images
# 2nd rows - images corrupted by noise
# 3rd rows - denoised images
rows, cols = 3, 9
num = rows * cols
imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_decoded[:num]])
imgs = imgs.reshape((rows * 3, cols, image_size, image_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          'Corrupted Input: middle rows, '
          'Denoised Input:  third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()






