import cv2
import numpy as np

#creating black and white boxes
black = np.zeros([500,500],dtype=np.uint8)
white = np.full([500,500],255,dtype=np.uint8)

row1 = cv2.hconcat([white,black])
row2 = cv2.hconcat([black,white])

img = cv2.imread('chess1.png')
cv2.imshow('img',img)






cv2.waitKey(0)
cv2.destroyAllWindows()