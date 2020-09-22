#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:47:14 2020

@author: pavankunchala
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_wid_plot(img,title):
    
    img_RGB= img[:,:,::-1]
    plt.imshow(img_RGB)
    plt.title(title)
    plt.show()
    
image = cv2.imread('img.png')

#showing the loaded imag
show_wid_plot(image, "Orignal image")

dst_img = cv2.resize(image,None,fx = 0.5, fy = 0.5 , interpolation = cv2.INTER_AREA)

height, width = image.shape[:2]

dst_image_2 = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)

show_wid_plot(dst_img, ' Reside img 1')
show_wid_plot(dst_image_2, ' Reside img 2')

M = np.float32([[1, 0, 200], [0, 1, 30]])

dst_image = cv2.warpAffine(image, M, (width, height))

show_wid_plot(dst_image, ' Translated image (+VE')

M = np.float32([[1, 0, -200], [0, 1, -30]])
dst_image = cv2.warpAffine(image, M, (width, height))

show_wid_plot(dst_image, ' Translated image -VE')

#roataion of the img
M = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 180, 1)
dst_image = cv2.warpAffine(image, M, (width, height))

cv2.circle(dst_image, (round(width / 2.0), round(height / 2.0)), 5, (255, 0, 0), -1)

show_wid_plot(dst_image, ' IMG rotated 180')

M = cv2.getRotationMatrix2D((width / 1.5, height / 1.5), 180, 1)
dst_image = cv2.warpAffine(image, M, (width, height))

cv2.circle(dst_image, (round(width / 1.5), round(height / 1.5)), 5, (255, 0, 0), -1)

show_wid_plot(dst_image, ' IMG rotated 30')







