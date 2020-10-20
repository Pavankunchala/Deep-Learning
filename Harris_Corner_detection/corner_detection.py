import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('chess.png')

image_copy = np.copy(image)

gray = cv2.cvtColor(image_copy,cv2.COLOR_BGR2GRAY)

#convert to float point
gray = np.float32(gray)

#detect the cornerns
dst= cv2.cornerHarris(gray,2,3,0.04)

#dilate the corners to enhanc it better
dst = cv2.dilate(dst,None)

plt.imshow(dst,cmap='gray')

#select and display strong contorsz
thresh = 0.01* dst.max()

corner_img = np.copy(image_copy)

for j in range(0,dst.shape[0]):
    for i in range(0,dst.shape[1]):
        if (dst[j,i]> thresh):
            cv2.circle(corner_img,(i,j),2,(0,255,0),1)
plt.imshow(corner_img)

plt.show()

