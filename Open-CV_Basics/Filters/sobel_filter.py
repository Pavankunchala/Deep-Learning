import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img.png',0)

#let's apply the sobel filter 

# Apply sobel filter along x direction
sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
# Apply sobel filter along y direction
sobely = cv2.Sobel(img,cv2.CV_32F,0,1)

# Normalize image for display
cv2.normalize(sobelx, 
                dst = sobelx, 
                alpha = 0, 
                beta = 1, 
                norm_type = cv2.NORM_MINMAX, 
                dtype = cv2.CV_32F)
cv2.normalize(sobely, 
                dst = sobely, 
                alpha = 0, 
                beta = 1, 
                norm_type = cv2.NORM_MINMAX, 
                dtype = cv2.CV_32F)

plt.figure(figsize=[10,6])
plt.subplot(121);plt.imshow(sobelx, cmap='gray');plt.title("Sobel X Gradients")
plt.subplot(122);plt.imshow(sobely, cmap='gray');plt.title("Sobel Y Gradients")
plt.show()