import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('road.png')

#convert grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#let's blur the image to reduce the noise
img_blur = cv2.medianBlur(gray,5)
 #Apply hough transform on the image
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 50, param1=450, param2=10, minRadius=30, maxRadius=40)
# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
plt.imshow(img[:,:,::-1])
plt.show()
