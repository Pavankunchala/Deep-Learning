from re import S
from numpy.core.defchararray import encode
import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, Input , Flatten
from tensorflow.keras.layers import Reshape , Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import os

"""
This Autoencoder will colorize the grayscale images 
We will be adding noise (Color) to the image such that it can colorize the image eventually
after it get's Trained 

"""

"""
First lets convert the color images to gray for training purposes

"""
def rgb2Gray(rgb):
    return np.dot(rgb[...,:3],[0.298,0.587,0.114])

#load the dataset
(x_train, _), (x_test, _) = cifar10.load_data()

#let's see the image dimensions
img_rows = x_train[1]
img_cols = x_train[2]
channels = x_train[3]

# create saved_images folder
imgs_dir = 'saved_images'
save_dir = os.path.join(os.getcwd(), imgs_dir)
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        

# display the 1st 100 input images (color and gray)
imgs = x_test[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test color images (Ground  Truth)')
plt.imshow(imgs, interpolation='none')
plt.savefig('%s/test_color.png' % imgs_dir)
plt.show()


"""
Now lets convert the train and test images to grayscale 
"""
x_train_gray = rgb2Gray(x_train)
x_test_gray = rgb2Gray(x_test)

# display grayscale version of test images
imgs = x_test_gray[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test gray images (Input)')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('%s/test_gray.png' % imgs_dir)
plt.show()

#normalize the color images
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#normalize the grayscale images
x_train_gray = x_train_gray.astype('float32') / 255
x_test_gray = x_test_gray.astype('float32') / 255

#reshape the color images in the format of row x col x channels
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)

#reshape the grayscale images in the format of row x col x channels
x_train_gray = x_train_gray.reshape(x_train_gray.shape[0], img_rows, img_cols, 1)
x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], img_rows, img_cols, 1)

# network parameters
input_shape = (img_rows, img_cols, 1)
batch_size = 32
kernel_size = 3
latent_dim = 256

layer_filters = [64, 128, 256]

"""
Let's start with the Autoencoder model 
First we will start with encoder model

"""
inputs = Input(shape = input_shape, name="encoder_input")

x = inputs

# stack of Conv2D(64)-Conv2D(128)-Conv2D(256)

for filters in layer_filters:
    
    x = Conv2D(filters = filters,kernel_size =kernel_size, strides=2,
               activation = 'relu', padding = 'same')(x)
    



"""
we need the shape info for building decoder model

"""
shape = K.int_shape(x)

#generate the latten vector
x = Flatten()(x)

latent = Dense(latent_dim, name = 'latent_vector')(x)

#instantiate the encoder model
encoder = Model(inputs , latent , name = 'encoder')
encoder.summary()

"""
Let's start the decoder model 
"""

latent_inputs = Input(shape = (latent_dim,), name = 'decoder_input')
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# stack of Conv2DTranspose(256)-Conv2DTranspose(128)-Conv2DTranspose(64)

for filters in layer_filters[::-1]:
    x  = Conv2DTranspose(filters = filters, kernel_size =kernel_size,
                         strides = 2,activation = 'relu', padding='same')(x)
    
    
    
outputs = Conv2DTranspose(filters=channels,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()



# autoencoder = encoder + decoder
# instantiate autoencoder model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()



# prepare model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'colorized_AutoEncoder_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

"""
now lets reduce the learning rate if it loss doesnt decrease for 5 epochs

"""
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown =0 , 
                               patience = 5, verbose= 2 ,min_lr = 0.5e -6 )


# save weights for future use (e.g. reload parameters w/o training)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=2,
                             save_best_only=True)

autoencoder.compile(loss='mse', optimizer='adam')

# called every epoch
callbacks = [lr_reducer, checkpoint]


# train the autoencoder
autoencoder.fit(x_train_gray,
                x_train,
                validation_data=(x_test_gray, x_test),
                epochs=40,
                batch_size=batch_size,
                callbacks=callbacks)

# predict the autoencoder output from test data
x_decoded = autoencoder.predict(x_test_gray)

# display the 1st 100 colorized images
imgs = x_decoded[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Colorized test images (Predicted)')
plt.imshow(imgs, interpolation='none')
plt.savefig('%s/colorized.png' % imgs_dir)
plt.show()





    


