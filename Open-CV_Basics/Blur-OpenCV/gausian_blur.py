import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('img.png')

# Apply gaussian blur
dst1=cv2.GaussianBlur(img,(5,5),0,0)
dst2=cv2.GaussianBlur(img,(25,25),50,50)

lineType=4
fontScale=1

# Display images


combined = np.hstack((img, dst1,dst2))
plt.figure(figsize=[10,6])

plt.subplot(131);plt.imshow(img[...,::-1]);plt.title("Original Image")
plt.subplot(132);plt.imshow(dst1[...,::-1]);plt.title("Gaussian Blur Result 1 : KernelSize = 5")
plt.subplot(133);plt.imshow(dst2[...,::-1]);plt.title("Gaussian Blur Result 2 : KernelSize = 25")