import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img.png')

scaleFactor = 1/255.0

img = np.float32(img)

#scale the values so that it lies in between [0,1]
img = img * scaleFactor

img = img * (1/scaleFactor)
img = np.uint8(img)

contrastPercentage = 30

contrastimg = img *(1 + contrastPercentage/100)

clippedContrastimg = np.clip(contrastimg,0,255)

clippedConvert = np.uint8(clippedContrastimg)

cv2.imwrite("ContrastIMG.png",clippedConvert)


