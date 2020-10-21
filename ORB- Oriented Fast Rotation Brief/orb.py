import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy


# Set the default figure size

image = cv2.imread("face.png")
query_image = cv2.imread("Group.jpg")

image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
query_image = cv2.cvtColor(query_image,cv2.COLOR_BGR2RGB)


training_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
query_gray = cv2.cvtColor(query_image,cv2.COLOR_RGB2GRAY)


#creating a orb
orb = cv2.ORB_create(1000,2.0)

#keypoingts and descriptoe
keypoints_train ,descriptors_train  = orb.detectAndCompute(training_gray,None)

keypoints_query ,descriptors_query  = orb.detectAndCompute(query_gray,None)

#create a brutr forece mathcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Perform the matching between the ORB descriptors of the training image and the query image
matches = bf.match(descriptors_train, descriptors_query)

# The matches with shorter distance are the ones we want. So, we sort the matches according to distance
matches = sorted(matches, key = lambda x : x.distance)

result = cv2.drawMatches(training_gray, keypoints_train, query_gray, keypoints_query,
                          matches[:100], query_gray, flags = 2)

plt.title('Best Matching Points')
plt.imshow(result)
plt.show()

# Print the number of keypoints detected in the training image
print("Number of Keypoints Detected In The Training Image: ", len(keypoints_train))

# Print the number of keypoints detected in the query image
print("Number of Keypoints Detected In The Query Image: ", len(keypoints_query))

# Print total number of matching points between the training and query images
print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))








