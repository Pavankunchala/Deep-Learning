import cv2
import matplotlib.pyplot as plt
import numpy as np

#let's start with contours
image = cv2.imread('thres.jpeg')

imageCopy = image.copy()

gray = cv2.cvtColor(image ,cv2.COLOR_BGR2GRAY)

#find all the contours 
contours , hierarchy = cv2.findContours(gray , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

#print("Number of contours found = {}".format(len(contours)))
#print("\nHierarchy : \n{}".format(hierarchy))

#draw the contours
#cv2.drawContours(image , contours ,-1 , (0,255,0),2)
#ontours = set(contours).to_list()
#contours = list(contours)
print(len(contours))
for cnt in contours:
    
    x,y,w,h = cv2.boundingRect(cnt)
    
    cv2.rectangle(image, (x-10,y-10), (x+w+10,y+h+10), (0,0,255), 2)

plt.imshow(image[:,:,::-1])
plt.show()