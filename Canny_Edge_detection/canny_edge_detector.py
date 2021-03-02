import matplotlib.pyplot as plt
import cv2
import numpy as np

image = cv2.imread('frames.png')

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
lower = 80
higher =100

#canny

edges = cv2.Canny(gray,lower,higher)
wide = cv2.Canny(gray,5,120)
#ight = cv2.Canny(gray,180,240)


#cv2.imshow("Edges",edges)


#cv2.imshow("Wide",wide)


#cv2.imshow("Tight",tight)


plt.imshow(edges)






plt.show()




