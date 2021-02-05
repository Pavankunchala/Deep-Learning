import cv2
import matplotlib.pyplot as plt

img = cv2.imread('img.png')

radius  = 75
center =(200,100)
new_img = img.copy()

circle_img = cv2.circle(img,center,radius,(200,0,100),thickness = 3 , lineType = cv2.LINE_AA)
circle_filled = cv2.circle(new_img,center,radius,(200,0,100),thickness = -1 , lineType = cv2.LINE_AA)

#converting BGR to RGB 

circle_img = circle_img[:, :, ::-1]
circle_filled = circle_filled[:,:,::-1]

plt.subplot(121)
plt.imshow(circle_img)
plt.subplot(122)
plt.imshow(circle_filled)
plt.show()