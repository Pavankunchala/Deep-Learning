import cv2
import matplotlib.pyplot as plt

img = cv2.imread('img.png')

#kernerl -== 3
dst1=cv2.blur(img,(3,3),(-1,-1))

# Apply box filter - kernel size 7
dst2=cv2.blur(img,(7,7),(-1,-1))

plt.figure(figsize=[10,6])
plt.subplot(131);plt.imshow(img[...,::-1]);plt.title("Original Image")
plt.subplot(132);plt.imshow(dst1[...,::-1]);plt.title("Box Blur Result 1 : KernelSize = 3")
plt.subplot(133);plt.imshow(dst2[...,::-1]);plt.title("Box Blur Result 2 : KernelSize = 7")
plt.show()