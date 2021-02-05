import cv2 as cv
import numpy as np
from tkinter import *

# Read image
im = cv.imread("blobs.jpg", cv.IMREAD_GRAYSCALE)

# Setup SimpleBlobDetector parameters.
params = cv.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255


#Filter by Area.
params.filterByArea = True
params.minArea = 200

"""
#Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5

# Filter by Convexity
params.filterByConvexity = 1
params.minConvexity = 0.98
params.maxConvexity = 1



# Filter by Inertia
params.filterByInertia = 1
params.minInertiaRatio = 0.0001
params.maxInertiaRatio = 1
"""

# Create a detector with the parameters
ver = (cv.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv.SimpleBlobDetector(params)
else:
    detector = cv.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                      cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('Filter by Area', im_with_keypoints)
cv.waitKey(0)
cv.destroyAllWindows()