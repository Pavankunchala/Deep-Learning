#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 09:29:25 2020

@author: pavankunchala
"""

import cv2
import dlib
import matplotlib.pyplot as plt

def show_img_with_matplotlib(color_img,title,pos):
    
    img_RGB = color_img[:,:,::-1]
    
    ax = plt.subplot(1,2,pos)
    
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_detection(image,faces):
    
    for face in faces:
        
        cv2.rectangle(image,(face.left(), face.top()),(face.right(),face.bottom()),(0,0,255),3)
        
    return image

img = cv2.imread('images.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#loading fronatal face detecor of dlib
detector = dlib.get_frontal_face_detector()

# Detect faces:
rects_1 = detector(gray, 0)

#here we are upsampling the img(increasing the size of the img) but takes more performance
rects_2 = detector(gray, 1)

#draw detections
img_faces_1 = show_detection(img.copy(), rects_1)
img_faces_2 = show_detection(img.copy(), rects_2)


fig = plt.figure(figsize=(20, 8))
plt.suptitle("Face detection using dlib frontal face detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(img_faces_1, "detector(gray, 0): " + str(len(rects_1)), 1)
show_img_with_matplotlib(img_faces_2, "detector(gray, 1): " + str(len(rects_2)), 2)

# Show the Figure:
plt.show()
        