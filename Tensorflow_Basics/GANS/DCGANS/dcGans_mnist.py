from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.layers import Dense , Activation , Conv2D , Flatten
from tensorflow.keras.layers import Input, Reshape , Conv2DTranspose, LeakyReLU , BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model , load_model
from tensorflow.keras.datasets import mnist

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse


"""
We are going to build a GAN (Generative Adversal Networks) to be more specific

We are going to build a DCGAN (Generative Adversarial Network (GAN) using CNN)

We have a generator and a discriminator

The generator tries to fool the discriminator by generating fake images.

The discriminator learns to discriminate real from fake images.

The generator + discriminator form an adversarial network.

DCGAN trains the discriminator and adversarial networks alternately.

While training  the discriminator not only learns how to differentiate between real and fake but also
gives valuable feedback to the generator and coaches it to generate better fake images.


"""

#Generator model

"""
Let's build the Generator part of the network
"""

def build_generator(inputs, image_size):
    
    image_resize = image_size //4
    
    #network parameters
    kernel_size =5
    layer_filters = [128, 64, 32, 1]
    
    x = Dense(image_resize * image_resize * layer_filters[0])(inputs)
    x = Reshape((image_resize , image_resize , layer_filters[0]))(x)
    
    for filters in layer_filters:
        # for first 2 filters stride  = 2
        #after that stride  = 1
       
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1
            
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters =filters, strides = strides,
                            kernel_size = kernel_size,padding = 'same')(x)
        
    x = Activation('sigmoid')(x)
    generator = Model(inputs , x , name = 'generator')
    return generator

"""
Let's build the discriminator part of the Adversarial network

"""

def build_discriminator(inputs):
    
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]
    
    x = inputs
    
    for filters in layer_filters:
        #for first 3 layers strides = 2 
        #else  = 1
        
        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides =2
            
        x = LeakyReLU(alpha = 0.2)(x)
        x = Conv2D(filters = filters,kernel_size =kernel_size,
                   strides = strides, padding = 'same')(x)
        
        
    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    
    discriminator = Model(inputs , x , name = 'discriminator')
    
    return discriminator


"""
Now let's train the model
 Alternately train Discriminator and Adversarial networks by batch.
    Discriminator is trained first with properly real and fake images.
    Adversarial is trained next with fake images pretending to be real
    Generate sample images per save_interval.

"""

def train(models, x_train,params):
    
    generator, discriminator , adversarial = models
    
    # the network parameters
    batch_size, latent_size, train_steps, model_name = params
    
    #save your generator image at every 500 epochs
    save_interval = 500
    
    # noise vector to see how the generator output evolves during training
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    # number of elements in train dataset
    train_size = x_train.shape[0]
    
    for i in range(train_steps):
        """
        train the discriminator for 1 batch
        1 batch of real (label=1.0) and fake images (label=0.0)
        randomly pick real images from dataset
        """
        
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        
        real_images = x_train[rand_indexes]
        
        #now let's generate some fake images from the generator using noise
        
        noise = np.random.uniform(-1.0,      1.0,
                                  size=[batch_size, latent_size])
        
        #generate some fake images
        fake_images = generator.predict(noise)
        
        #training data = real + fake = 1 batch
        x = np.concatenate((real_images, fake_images))
        
        #we have already decided that real images label is 1.0 
        y = np.ones([2 * batch_size, 1])
        
        #and that of fake images is 0
        y[batch_size:, :] = 0.0
        
        #now lets train the discriminator network and log the loss and error
        
        loss, acc  = discriminator.train_on_batch(x,y)
        
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)
        
        """
        Now let's train the adversarial network
         1 batch of fake images with label=1.0
         since the discriminator weights 
         are frozen in adversarial network
         only the generator is trained
        generate noise using uniform distribution
        """
        noise = np.random.uniform(-1.0,    1.0, 
                                  size=[batch_size, latent_size])
        
        #now let's label the fake images as real ( 1.0)
        
        y = np.ones([batch_size , 1])
        
        """
        Now let's train the discriminator with the fake images
        the fake images go to the discriminator input of the adversarial
        for classification
        """     
        loss, acc = adversarial.train_on_batch(noise, y)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        
        print(log)
        
        if (i + 1) % save_interval == 0:
            
            #plot the images on a regular basis 
            
            plot_images(generator,
                        noise_input=noise_input,
                        show=False,
                        step=(i + 1),
                        model_name=model_name)
            
            
        #now let's save the model for future Mnist Generation
        
    generator.save(model_name + ".h5")
    
    
"""
Now let's generate some fake images and plot them

"""
    
    
def plot_images(generator,noise_input,
                show = False, step = 0 , model_name = 'GAN_mnist'):
    
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    
    images = generator.predict(noise_input)
    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    
    if show:
        plt.show()
    else:
        plt.close('all')
        
"""
Now let's build and train the model
"""

def build_and_train_models():
    
    (x_train, _), (_, _) = mnist.load_data()
    
    # reshape data for CNN as (28, 28, 1) and normalize
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    
    model_name = 'dcGAN_mnist'
    
    # network parameters
    # the latent or z vector is 100-dim
    latent_size = 100
    batch_size = 64
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    
    #build discriminator model
    inputs = Input(shape = input_shape , name = 'discriminator_input')
    
    discriminator = build_discriminator(inputs)
    
    optimizer = RMSprop(lr = lr , decay = decay )
    discriminator.compile(loss = 'binary_crossentropy', metrics=['accuracy'],
                          optimizer = optimizer)
    
    #let's see the summary 
    discriminator.summary()
    
    #now let's  build the generator model
    
    input_shape = (latent_size, )
    
    inputs = Input(shape=input_shape, name='z_input')
    generator = build_generator(inputs, image_size)
    generator.summary()
    
    #now let's build the adversarial model'
    
    optimizer = RMSprop(lr=lr * 0.5, decay=decay * 0.5)
    #we will free the weighrs of discriminator during adversarial network
    
    discriminator.trainable = False
    
    # adversarial = generator + discriminator
    adversarial = Model(inputs, 
                        discriminator(generator(inputs)),
                        name=model_name)
    
    adversarial.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()
    
    # train discriminator and adversarial networks
    models = (generator, discriminator, adversarial)
    params = (batch_size, latent_size, train_steps, model_name)
    train(models, x_train, params)
    
    
def test_generator(generator):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    plot_images(generator,
                noise_input=noise_input,
                show=True,
                model_name="test_outputs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        test_generator(generator)
    else:
        build_and_train_models()
    
    
    
    
        
        
    
    
        
    
            
            
        
           
        
        
        
        
        
    
    
    
    
    




    
    
    

    
        
        
    