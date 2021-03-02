import cv2
import matplotlib.pyplot as plt   
import numpy as np

img = cv2.imread('img.png')
hsvImg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


#splitting hSV 
H ,S, V = cv2.split(hsvImg)

#let's remove the white or gray pixels

H_array = H[S > 10].flatten()


plt.subplot(121);plt.imshow(img[...,::-1]);plt.title("Image");plt.axis('off')
plt.subplot(122);plt.hist(H_array, bins=180, color='r');plt.title("Histogram")
plt.show()


