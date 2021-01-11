# Python3 program change RGB Color 
# Model to HSV Color Model 

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('img.png')



b,g,r = cv2.split(img)
r  = 0.299*r 
g  =0.587*g
b = 0.114*b
result =r+g+b



gray = cv2.merge([b,g,r])
plt.imshow(gray)
plt.show()


