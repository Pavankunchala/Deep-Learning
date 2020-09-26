#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 10:33:55 2020

@author: pavankunchala
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import collections
import dlib


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

def color_quantization(image , k):
    
    #transforming image into data
    data= np.float32(image).reshape((-1,3))
    
    # Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
    # In this case the maximum number of iterations is set to 20 and epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS +cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    
    # Apply K-means clustering algorithm:
    ret, label , center = cv2.kmeans(data, k,  None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
     # Convert center to uint8:
    center = np.uint8(center)
    
    #replacing the pixel values with center values
    result = center[label.flatten()]
    result = result.reshape(image.shape)
    
    counter = collections.Counter(label.flatten())
    print(counter)
    
    # Calculate the total number of pixels of the input image:
    total = image.shape[0] * image.shape[1]
    
     # Assign width and height to the color_distribution image:
    desired_width = image.shape[1]
    # The difference between 'desired_height' and 'desired_height_colors'
    # will be the separation between the images
    desired_height = 70
    desired_height_colors = 50

    # Initialize the color_distribution image:
    color_distribution = np.ones((desired_height, desired_width, 3), dtype="uint8") * 255
    # Initialize start:
    start = 0

    for key, value in counter.items():
        # Calculate the normalized value:
        value_normalized = value / total * desired_width

        # Move end to the right position:
        end = start + value_normalized

        # Draw rectangle corresponding to the current color:
        cv2.rectangle(color_distribution, (int(start), 0), (int(end), desired_height_colors), center[key].tolist(), -1)
        # Update start:
        start = end


    return np.vstack((color_distribution, result))

fig = plt.figure(figsize=(16, 8))
plt.suptitle("Color quantization using K-means clustering algorithm", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load BGR image:
img = cv2.imread('photo.jpeg')

# Apply color quantization:
color_3 = color_quantization(img, 3)
color_5 = color_quantization(img, 5)
color_10 = color_quantization(img, 10)
color_20 = color_quantization(img, 20)
color_40 = color_quantization(img, 40)

# Plot the images:
show_img_with_matplotlib(img, "original image", 1)
show_img_with_matplotlib(color_3, "color quantization (k = 3)", 2)
show_img_with_matplotlib(color_5, "color quantization (k = 5)", 3)
show_img_with_matplotlib(color_10, "color quantization (k = 10)", 4)
show_img_with_matplotlib(color_20, "color quantization (k = 20)", 5)
show_img_with_matplotlib(color_40, "color quantization (k = 40)", 6)

# Show the Figure:
plt.show()

