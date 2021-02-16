import cv2
import numpy as np 
import matplotlib.pyplot as plt

img = cv2.imread('sky.png')

#convert to HSV format
hsvIM = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
hsvIMco = hsvIM.copy()


# Perform histogram equalization only on the V channel
hsvIM[:,:,2] = cv2.equalizeHist(hsvIM[:,:,2])

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
hsvIMco[:,:,2] = clahe.apply(hsvIMco[:,:,2])

#convert back to BGR format
hsvIMeq = cv2.cvtColor(hsvIM, cv2.COLOR_HSV2BGR)
coEQ = cv2.cvtColor(hsvIMco, cv2.COLOR_HSV2BGR)

plt.figure(figsize=(10,6))

#cv2.imwrite('HistograEqualize.png',hsvIMeq)
#cv2.imwrite('CLAHE_histogram.png',coEQ)

ax = plt.subplot(1,3,1)
plt.imshow(img[:,:,::-1])
ax.set_title("Original Image")
ax.axis('off')


ax = plt.subplot(1,3,2)
plt.imshow(hsvIMeq[:,:,::-1])
ax.set_title("Histogram Equalized")
ax.axis('off')

ax = plt.subplot(1,3,3)
plt.imshow(coEQ[:,:,::-1])
ax.set_title("CLAHE")
ax.axis('off')

plt.show()


