
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('img.png')

 #Sharpen kernel
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

# Using 2D filter by applying the sharpening kernel
sharpenOutput = cv2.filter2D(img, -1, sharpen)

plt.figure(figsize=[10,6])

plt.subplot(121)
plt.imshow(img[:,:,::-1])
plt.title('Original Image')

plt.subplot(122)
plt.imshow(sharpenOutput[:,:,::-1])
plt.title('Sharpened')

plt.show()