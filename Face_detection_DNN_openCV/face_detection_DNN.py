#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 21:54:44 2020

@author: pavankunchala
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')
    

#loading the pre-trained model

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")

# Load image:
image = cv2.imread("team.jpg")

# Get dimensions 
(h, w) = image.shape[:2]

# Create 4-dimensional blob from image:
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104., 117., 123.], False, False)

# Set the blob as input and obtain the detections:
net.setInput(blob)
detections = net.forward()

# Initialize the number of detected faces counter detected_faces:
detected_faces = 0

# Iterate over all detections:
for i in range(0, detections.shape[2]):
    # Get the confidence (probability) of the current detection:
    confidence = detections[0, 0, i, 2]

    # Only consider detections if confidence is greater than a fixed minimum confidence:
    if confidence > 0.6:
       
        detected_faces += 1
       
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw the detection and the confidence:
        text = "{:.3f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 3)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 5))
plt.suptitle("Face detection using OpenCV DNN face detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(image, "DNN face detector: " + str(detected_faces), 1)

# Show the Figure:
plt.show()