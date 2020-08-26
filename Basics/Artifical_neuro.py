#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:10:35 2020

@author: pavankunchala
"""

import math

def sigmoid(x):
    
    
    y = 1.0 / (1 + math.exp(-x))
    return y

def activate(inputs, weights):
   
    h = 0

    # compute the sum of the product of the input signals and the weights
    # here we're using pythons "zip" function to iterate two lists together
    for x, w in zip(inputs, weights):
        h += x*w

    # process sum through sigmoid activation function
    return sigmoid(h)
if __name__ == "__main__":
    inputs = [0.5, 0.3, 0.2]
    weights = [0.4, 0.7, 0.2]
    output = activate(inputs, weights)
    print(output)
