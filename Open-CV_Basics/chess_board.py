import cv2
import numpy as np

#creating black and white boxes
black = np.zeros([500,500],dtype=np.uint8)
white = np.full([500,500],255,dtype=np.uint8)

row1 = cv2.hconcat([white,black])
row2 = cv2.hconcat([black,white])

cv2.imwrite("chess1.png",cv2.vconcat([row1,row2]))






cv2.waitKey(0)
cv2.destroyAllWindows()