import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img.png')

line_img = cv2.line(img,(200,80),(600,50) , (255,255,0), thickness= 3,lineType = cv2.LINE_AA)

plt.imshow(line_img[:,:,::-1])

plt.show()
