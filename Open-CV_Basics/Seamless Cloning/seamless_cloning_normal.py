import cv2
import numpy as np
import matplotlib.pyplot as plt

#Let's add the Source image (the image we will clone into the new image)
src = cv2.imread('pa.jpg')

#let's add the Destination image (the image which the source image be cloned into)
dst = cv2.imread('mon.jpg')

# Create a rough mask around the src image
src_mask =255* np.ones(src.shape, src.dtype)
#poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)*8
#src_mask = cv2.fillPoly(src_mask, [poly], (255, 255, 255))

width, height, channels = dst.shape
# This is where the center of the source image is kept
center = (746 ,720)

#clone seamlessly
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.MIXED_CLONE)

cv2.imwrite('out.png',output)

plt.imshow(output[:,:,::-1])
plt.show()