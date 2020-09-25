#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 08:21:58 2020

@author: pavankunchala
"""

# Import required packages:
import cv2
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """ Shows an image using matplotlib capabilities """

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(8, 6))
plt.suptitle("ORB descriptors and Brute-Force (BF) matcher", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

#query and scence
query = cv2.imread('img1.jpg')
scene = cv2.imread('img2.png')

#initating orb detecor
orb = cv2.ORB_create()

#detectors and keypointa

keypoints_1, descriptors_1 = orb.detectAndCompute(query, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(scene, None)

#BF Matcher 
#brute force marcher

bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#match  descriptors
bf_matches = bf_matcher.match(descriptors_1, descriptors_2)

# Sort the matches in the order of their distance:
bf_matches = sorted(bf_matches, key=lambda x: x.distance)

# Draw first 20 matches:
result = cv2.drawMatches(query, keypoints_1, scene, keypoints_2, bf_matches[:20], None,
                         matchColor=(255, 255, 0), singlePointColor=(255, 0, 255), flags=0)

# Plot the images:
show_img_with_matplotlib(result, "matches between the two images", 1)

# Show the Figure:
plt.show()


