#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 23:11:08 2020

@author: pavankunchala
"""

import numpy as np
import cv2
import matplotlib.pyplot as  plt

def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')
    
fig = plt.figure(figsize=(12, 5))
plt.suptitle("ORB keypoint detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = cv2.imread('img.png')

#initTING  the orb detectior
orb = cv2.ORB_create()
#detecting the key points using orb
keypoints = orb.detect(image,None)

#computing the keypouns and the descirptor
keypoints, descriptors = orb.compute(image, keypoints)

# Print one ORB descriptor:
print("First extracted descriptor: {}".format(descriptors[0]))

#drawinf keypoints
image_keypoints =cv2.drawKeypoints(image, keypoints, outImage = None,color=(255, 0, 255), flags=0)


# Plot the images:
show_img_with_matplotlib(image, "image", 1)
show_img_with_matplotlib(image_keypoints, "detected keypoints", 2)

# Show the Figure:
plt.show()