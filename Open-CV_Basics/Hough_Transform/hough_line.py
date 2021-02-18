import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('road.png')

#convert grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#let's find the edges in Canny Edge detector
edges = cv2.Canny(gray,200,250)

"""
So this is how a HoughLines work we need a set of points that can form a line 

"""

lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength = 240 , maxLineGap = 250)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    

cv2.imwrite('HoughLine.png',img)
    
plt.imshow(img[:,:,::-1])

plt.show()

