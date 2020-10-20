import matplotlib.pyplot as plt
import cv2
import numpy as np

image = cv2.imread('pika.jpg')

image_copy = np.copy(image)

#resizing the image
image_copy = cv2.resize(image_copy,(0,0),fx = 0.5, fy = 0.5)

#changing to RGB
image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)

#cv2.imshow("The original",image_copy)

#convert the image to Gray 
gray = cv2.cvtColor(image_copy,cv2.COLOR_RGB2GRAY)

#cv2.imshow("The gray one",gray)

#defining the lower and higher Thresholds for hysteries

lower = 120
higher =240

#canny

edges = cv2.Canny(gray,lower,higher)
wide = cv2.Canny(gray,30,100)
tight = cv2.Canny(gray,180,240)

#plt.subplot(131)
cv2.imshow("Edges",edges)
#plt.subplot(132)

cv2.imshow("Wide",wide)
#plt.subplot(133)

cv2.imshow("Tight",tight)



plt.show()




