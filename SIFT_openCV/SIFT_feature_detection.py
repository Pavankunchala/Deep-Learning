import cv2
import numpy as np


img = cv2.imread("build.jpg")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#creating the sift algo
sift  = cv2.SIFT_create()
kp = sift.detect(gray,None)
img=cv2.drawKeypoints(gray,kp,img)


cv2.imwrite('Sift_image.png',img)

