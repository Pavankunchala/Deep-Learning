import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('img.png',0)

kernelSize =3


 #Applying laplacian
img1 = cv2.GaussianBlur(img,(3,3),0,0)
laplacian = cv2.Laplacian(img1, cv2.CV_32F, ksize = kernelSize, 
                            scale = 1, delta = 0)
cv2.normalize(laplacian, 
                dst = laplacian, 
                alpha = 0, 
                beta = 1, 
                norm_type = cv2.NORM_MINMAX, 
                dtype = cv2.CV_32F)

plt.figure(figsize=[10,6])
plt.imshow(laplacian,cmap='gray');plt.title("Laplacian")
plt.show()