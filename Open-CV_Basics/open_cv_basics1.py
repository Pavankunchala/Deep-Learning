#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:08:06 2020

@author: pavankunchala
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("img.png")

#converting the image from one format to another
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
b, g, r = cv2.split(image)
different_img = cv2.merge([r,g,b])



plt.subplot(131)
cv2.imshow("The orignal image", image)
plt.subplot(132)
cv2.imshow("The gray version", gray_image)
plt.subplot(133)
cv2.imshow("The different colorverison", different_img)


plt.show()




cv2.waitKey(0)

cv2.destroyAllWindows()

