#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:45:32 2020

@author: pavankunchala
"""

import cv2
import matplotlib.pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    """ Shows an image using matplotlib capabilities """

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

fig = plt.figure(figsize=(12, 5))
plt.suptitle("Aruco markers creation", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

#crearting an aruco dictionary
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)

#draw marker
aruco_marker_1  =cv2.aruco.drawMarker(dictionary= aruco_dictionary, id = 2 , 
                                      sidePixels=600, borderBits=1)
aruco_marker_2  =cv2.aruco.drawMarker(dictionary= aruco_dictionary, id = 2 , 
                                      sidePixels=600, borderBits=2)
aruco_marker_3  =cv2.aruco.drawMarker(dictionary= aruco_dictionary, id = 2 , 
                                      sidePixels=600, borderBits=3)

cv2.imwrite("marker_DICT_7X7_250_600_1.png", aruco_marker_1)
cv2.imwrite("marker_DICT_7X7_250_600_2.png", aruco_marker_2)
cv2.imwrite("marker_DICT_7X7_250_600_3.png", aruco_marker_3)

# Plot the images:
show_img_with_matplotlib(cv2.cvtColor(aruco_marker_1, cv2.COLOR_GRAY2BGR), "marker_DICT_7X7_250_600_1", 1)
show_img_with_matplotlib(cv2.cvtColor(aruco_marker_2, cv2.COLOR_GRAY2BGR), "marker_DICT_7X7_250_600_2", 2)
show_img_with_matplotlib(cv2.cvtColor(aruco_marker_3, cv2.COLOR_GRAY2BGR), "marker_DICT_7X7_250_600_3", 3)

# Show the Figure:
plt.show()
