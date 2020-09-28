#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:43:51 2020

@author: pavankunchala
"""

import cv2
import cvlib as cv
import matplotlib.pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_detection(image, faces):
    """Draws a rectangle over each detected face"""

    for (startX, startY, endX, endY) in faces:
        cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)

    return image


image = cv2.imread('img.jpg')

# Detect faces:
faces, confidences = cv.detect_face(image)

#Draw face detections:
img_result = show_detection(image.copy(), faces)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 5))
plt.suptitle("Face detection using cvlib face detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(img_result, "cvlib face detector: " + str(len(faces)), 1)

# Show the Figure:
plt.show()





