import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread("phone.jpg")

#converting the image 
image_copy = np.copy(image)

image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)

plt.imshow(image_copy)

plt.imshow(image)

#conver the image to gray
gray = cv2.cvtColor(image_copy,cv2.COLOR_RGB2GRAY)

low_threshold = 50
high_threshold = 120
edges = cv2.Canny(gray,low_threshold,high_threshold)

plt.imshow(edges,cmap='gray')



#parameters of Hough transform
rho = 1
thetha = np.pi/180
threshold = 60
max_line_length= 75
max_line_gap = 5

#find the lines using hough transform
lines = cv2.HoughLinesP(edges,rho,thetha,threshold,np.array([]),
                        max_line_length,max_line_gap)

line_image = np.copy(image_copy)

for line in lines:
    
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
        
plt.imshow(line_image)
plt.show()


        






