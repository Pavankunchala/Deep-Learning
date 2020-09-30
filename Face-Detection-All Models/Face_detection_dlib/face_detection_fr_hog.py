#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:11:56 2020

@author: pavankunchala
"""

import cv2
import face_recognition
import matplotlib.pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')
    
def show_detection(image, faces):
    
    for face in faces:
        top,right,bottom,left = face
        
        image = cv2.rectangle(image,(left, top),(right, bottom),(0,0,255),3)
        
    return image

img = cv2.imread('images.jpeg')

# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
rgb = img[:, :, ::-1]

rects_1 = face_recognition.face_locations(rgb,2,'hog')
rects_2 = face_recognition.face_locations(rgb,3,'hog')
#if u want CNN
rects_3 = face_recognition.face_locations(rgb,2,'cnn')
rects_4 = face_recognition.face_locations(rgb,3,'cnn')


# Draw face detections:
img_faces_1 = show_detection(img.copy(), rects_1)
img_faces_2 = show_detection(img.copy(), rects_2)
img_faces_3 = show_detection(img.copy(), rects_3)
img_faces_4 = show_detection(img.copy(), rects_4)



# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 4))
plt.suptitle("Face detection using face_recognition frontal face detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(img_faces_1, "face_locations(rgb, 0, hog): " + str(len(rects_1)), 1)
show_img_with_matplotlib(img_faces_2, "face_locations(rgb, 1, hog): " + str(len(rects_2)), 2)
show_img_with_matplotlib(img_faces_3, "face_locations(rgb, 0, cnn): " + str(len(rects_3)), 3)
show_img_with_matplotlib(img_faces_4, "face_locations(rgb, 1, cnn): " + str(len(rects_4)), 4)


# Show the Figure:
plt.show()
