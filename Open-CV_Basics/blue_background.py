import  matplotlib.pyplot as plt
import cv2
import numpy as np


image = cv2.imread('ic.jpg')
image_copy = np.copy(image)

image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)



lower_blue = np.array([0,0,200])
upper_blue = np.array([50,70,255])

mask = cv2.inRange(image_copy,lower_blue,upper_blue)



masked_image = np.copy(image_copy)
masked_image[mask !=0] = [0,0,0]
back_ground_img = cv2.imread('space.jpg')
back_ground_img = cv2.cvtColor(back_ground_img,cv2.COLOR_BGR2RGB)

#crop the imgh to 640 * 360
crop_background = back_ground_img[0:360,0:640]

crop_background[mask ==0] = [0,0,0]

complete_img = crop_background + masked_image

cv2.imshow("the complete image",complete_img)




plt.subplot(121)
plt.imshow(image_copy)
plt.subplot(122)
plt.imshow(masked_image)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()




#adding the backgroud img



