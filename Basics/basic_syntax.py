#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 12:30:40 2020

@author: pavankunchala
"""

import tensorflow as tf
#checking the version

print(tf.__version__)


hello = tf.constant(20)
world = tf.constant(30)

#just checking the types


print(type(hello))
#just checking the string

print(hello)

with tf.compat.v1.Session() as sess:
    
    
    result = sess.run(hello + world)
    
print(result)    
    


