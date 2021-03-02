import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('img.png')

# diameter of the pixel neighbourhood used during filtering
dia=15

# Larger the value the distant colours will be mixed together 
# to produce areas of semi equal colors
sigmaColor=80

# Larger the value more the influence of the farther placed pixels 
# as long as their colors are close enough
sigmaSpace=80

#Apply bilateralFilter
result = cv2.bilateralFilter(img, dia, sigmaColor, sigmaSpace)

plt.figure(figsize=[10,6])
plt.subplot(121);plt.imshow(img[...,::-1]);plt.title("Original Image")
plt.subplot(122);plt.imshow(result[...,::-1]);plt.title("Bilateral Blur Result")
plt.show()

"""
Median filtering is the best way to smooth images which have salt-pepper type of noise (sudden high / low values in the neighborhood of a pixel).

Gaussian filtering can be used if there is low Gaussian noise.

Bilateral Filtering should be used if there is high level of Gaussian noise, and you want the edges intact while blurring other areas.

In terms of execution speed, Gaussian filtering is the fastest and Bilateral filtering is the slowest.
"""
