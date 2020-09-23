#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:02:04 2020

@author: pavankunchala
"""
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('img.jpg')

def show_with_matplotlib(color_img,title,pos):
    
    img_RGB = color_img[:,:,::-1]
    
    ax = plt.subplot(2,4,pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis("off")
    
def sketch_img(img):
    
    #convert to gray_scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #applying median filter
    img_gray = cv2.medianBlur(img_gray,5)
    
    #detecting the images
    edges = cv2.Laplacian(img_gray,cv2.CV_8U,ksize =5)
    
    #threhold for images
    ret, thresholded = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
    
    return thresholded

def cartoonize_image(img, gray_mode = False):
    
    thresholded = sketch_img(img)
    
    #applying bilateral fliter wid big numbers to get cartoonnized
    filtered= cv2.bilateralFilter(img,10,250,250)
    
    cartoonized = cv2.bitwise_and(filtered, filtered, mask=thresholded)
    
    if gray_mode:
        return cv2.cvtColor(cartoonized, cv2.COLOR_BGR2GRAY)
    
    return cartoonized

plt.figure(figsize=(14, 6))
plt.suptitle("Cartoonizing images", fontsize=14, fontweight='bold')

# Call the created functions for sketching and cartoonizing images:
custom_sketch_image = sketch_img(image)
custom_cartonized_image = cartoonize_image(image)
custom_cartonized_image_gray = cartoonize_image(image, True)


sketch_gray, sketch_color = cv2.pencilSketch(image, sigma_s=30, sigma_r=0.1, shade_factor=0.1)
stylizated_image = cv2.stylization(image, sigma_s=60, sigma_r=0.07)


show_with_matplotlib(image, "image", 1)
show_with_matplotlib(cv2.cvtColor(custom_sketch_image, cv2.COLOR_GRAY2BGR), "custom sketch", 2)
show_with_matplotlib(cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR), "sketch gray cv2.pencilSketch()", 3)
show_with_matplotlib(sketch_color, "sketch color cv2.pencilSketch()", 4)
show_with_matplotlib(stylizated_image, "cartoonized cv2.stylization()", 5)
show_with_matplotlib(custom_cartonized_image, "custom cartoonized", 6)
show_with_matplotlib(cv2.cvtColor(custom_cartonized_image_gray, cv2.COLOR_GRAY2BGR), "custom cartoonized gray", 7)

# Show the created image:
plt.show()





    
    
              




