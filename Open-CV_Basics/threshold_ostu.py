#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:25:36 2020

@author: pavankunchala
"""

import cv2
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_hist_with_matplotlib_gray(hist, title, pos, color, t=-1):
    """Shows the histogram using matplotlib capabilities"""

    ax = plt.subplot(3, 2, pos)
    # plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.axvline(x=t, color='m', linestyle='--')
    plt.plot(hist, color=color)


# Create the dimensions of the figure and set title and color:
fig = plt.figure(figsize=(11, 10))
plt.suptitle("Otsu's binarization algorithm applying a Gaussian filter", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load the image and convert it to grayscale:
image = cv2.imread('img.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate the histogram
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Otsu's binarization algorithm:
ret1, th1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#  Blurs the image using a Gaussian filter to eliminate noise
gray_image_blurred = cv2.GaussianBlur(gray_image, (25, 25), 0)

# Calculate histogram after filtering:
hist2 = cv2.calcHist([gray_image_blurred], [0], None, [256], [0, 256])

# Otsu's binarization algorithm:
ret2, th2 = cv2.threshold(gray_image_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Plot all the images:
show_img_with_matplotlib(image, "image with noise", 1)
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img with noise", 2)
show_hist_with_matplotlib_gray(hist, "grayscale histogram", 3, 'm', ret1)
show_img_with_matplotlib(cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR),
                         "Otsu's binarization (before applying a Gaussian filter)", 4)
show_hist_with_matplotlib_gray(hist2, "grayscale histogram", 5, 'm', ret2)
show_img_with_matplotlib(cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR),
                         "Otsu's binarization (after applying a Gaussian filter)", 6)

# Show the Figure:
plt.show()