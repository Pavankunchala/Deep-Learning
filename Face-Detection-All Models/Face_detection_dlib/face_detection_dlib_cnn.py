#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:58:51 2020

@author: pavankunchala
"""

import cv2
import matplotlib.pyplot as plt
import dlib

def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')
    
def show_detection(image,faces):
    
    for face in faces:
        cv2.rectangle(image, (face.rect.left(), face.rect.top()), (face.rect.right(), 
                                                                   face.rect.bottom()), (0, 0, 255),  5)
    return image


#loading cnn detecor from dlib
cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

img = cv2.imread('team.jpg')
# Resize the image to attain reasonable speed:
#img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

# Detect faces:
rects = cnn_face_detector(img, 2)

# Draw face detections:
img_faces = show_detection(img.copy(), rects)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 5))
plt.suptitle("Face detection using dlib CNN face detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(img_faces, "cnn_face_detector(img, 0): " + str(len(rects)), 1)

# Show the Figure:
plt.show()




