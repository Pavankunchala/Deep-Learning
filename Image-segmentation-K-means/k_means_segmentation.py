import cv2
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread("butter.jpg")

image_copy = np.copy(image)

#convert to rgb
image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)

plt.imshow(image_copy)

#prep data for k-means
pixel_vals = image_copy.reshape((-1,3))

#convet the float type
pixel_vals = np.float32(pixel_vals)

#defining the criteria
criteria  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

#implement k - means
k = 4
retval,labels ,centers = cv2.kmeans(pixel_vals,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

#convert the data into 8 bit values
centers = np.uint8(centers)
segemented_data = centers[labels.flatten()]

#reshape
segemented_image = segemented_data.reshape((image_copy.shape))
labels_reshape = labels.reshape(image_copy.shape[0],image_copy.shape[1])

plt.title("K = 4")
plt.imshow(segemented_image)

#plt.imshow(labels_reshape==1, cmap='gray')



plt.show()
