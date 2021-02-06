import cv2
import matplotlib.pyplot as plt
import numpy as np

#let's start with contours
image = cv2.imread('Contour.png')

imageCopy = image.copy()

gray = cv2.cvtColor(image ,cv2.COLOR_BGR2GRAY)

#find all the contours 
contours , hierarchy = cv2.findContours(gray , cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)

print("Number of contours found = {}".format(len(contours)))
print("\nHierarchy : \n{}".format(hierarchy))

#draw the contours
cv2.drawContours(image , contours ,-1 , (0,255,0),2)

plt.imshow(image[:,:,::-1])
#plt.show()


#now lets only find external Contours 
contours , hierarchy = cv2.findContours(gray , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

print("Number of contours found = {}".format(len(contours)))

image = imageCopy.copy()
cv2.drawContours(image , contours ,-1 , (0,255,255),2)
plt.imshow(image[:,:,::-1])
#plt.show()

#Now lets try to draw only a single contour in that 

image = imageCopy.copy()
cv2.drawContours(image , contours[3] ,-1 , (0,255,255),4)

plt.imshow(image[:,:,::-1])
#plt.show()

"""
Finding contours is only the basic part we can use this information for many other things
Such as Object detection and Recognition 
Now for basics lets try to find the centroids of the contours

"""
image = imageCopy.copy()

contours , hierarchy = cv2.findContours(gray , cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image , contours ,-1 , (0,255,0),2)

#lets find the centroids
for cnt in contours:
    M = cv2.moments(cnt)
    x = int(round(M["m10"]/M["m00"]))
    y = int(round(M["m01"]/M["m00"]))
    
    # Mark the center
    cv2.circle(image, (x,y), 10, (0,0,255), -1)
    
plt.imshow(image[:,:,::-1])
#plt.show()

"""
Let's Add bounding boxes for the contours

"""
#Verticle rectangle (overlapping might happen)
image = imageCopy.copy()

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x,y), (x+w,y+h), (255,130,255), 2)
   
plt.imshow(image[:,:,::-1])
#plt.show() 

#Roated Bounding boxes

image = imageCopy.copy()

for cnt in contours:
    box = cv2.minAreaRect(cnt)
    boxPts = np.int0(cv2.boxPoints(box))
    
    cv2.drawContours(image, [boxPts], -1, (0,0,255), 2)
    
plt.imshow(image[:,:,::-1])
plt.show()






    
    










