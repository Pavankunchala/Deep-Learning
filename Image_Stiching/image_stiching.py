#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 22:01:06 2020

@author: pavankunchala
"""

#importing stuff
import cv2
import numpy as np
import matplotlib.pyplot as plt

import imageio
import imutils
cv2.ocl.setUseOpenCL(False)

feature_extractor = 'orb' # one of 'sift', 'surf', 'brisk', 'orb'
feature_matching = 'bf'

img_i = cv2.imread('Original_left.png')
img_i = cv2.resize(img_i,(0,0),fx = 1,fy = 1)
img1 = cv2.cvtColor(img_i,cv2.COLOR_BGR2GRAY)

img_j = cv2.imread('Original_middle.png')
img_j= cv2.resize(img_j,(0,0),fx = 1,fy = 1)
img2 = cv2.cvtColor(img_j,cv2.COLOR_BGR2GRAY)

img_k= cv2.imread('Original_right.png')
img_k = cv2.resize(img_k,(0,0),fx = 1,fy = 1)
img3 = cv2.cvtColor(img_k,cv2.COLOR_BGR2GRAY)



fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, constrained_layout=False, figsize=(20,8))
ax1.imshow(img_i, cmap="gray")
ax1.set_xlabel("Query image", fontsize=14)

ax2.imshow(img_j, cmap="gray")
ax2.set_xlabel("Train image (Image to be transformed)", fontsize=14)


ax3.imshow(img_k, cmap="gray")
ax3.set_xlabel("Train image (Image to be transformed)", fontsize=14)

plt.show()


def detectAndDescribe(image, method=None):
   
    assert method is not None 
    
    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)

kpsA, featuresA = detectAndDescribe(img1, method=feature_extractor)
kpsB, featuresB = detectAndDescribe(img2, method=feature_extractor)
kpsC, featuresC = detectAndDescribe(img3, method=feature_extractor)



fig, (ax1,ax2, ax3) = plt.subplots(nrows=1, ncols =3, figsize=(20,8), constrained_layout=False)
ax1.imshow(cv2.drawKeypoints(img1,kpsA,None,color=(0,255,0)))
ax1.set_xlabel("(a)", fontsize=14)
ax2.imshow(cv2.drawKeypoints(img2,kpsB,None,color=(0,255,0)))
ax2.set_xlabel("(b)", fontsize=14)
ax3.imshow(cv2.drawKeypoints(img3,kpsC,None,color=(0,255,0)))
ax3.set_xlabel("(c)", fontsize=14)

plt.show()


def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"
    
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf


def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)
        
    # Match descriptors.
    best_matches = bf.match(featuresA,featuresB)
    
    # Sort the features in order of distance.
   
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches



def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches


print("Using: {} feature matcher".format(feature_matching))

fig = plt.figure(figsize=(20,8))

if feature_matching == 'bf':
    matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
    imgg = cv2.drawMatches(img1,kpsA,img2,kpsB,matches[:100],
                           None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
elif feature_matching == 'knn':
    matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
    imgg = cv2.drawMatches(img1,kpsA,img2,kpsB,np.random.choice(matches,100),
                           None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    

plt.imshow(imgg)
plt.show()


def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            reprojThresh)

        return (matches, H, status)
    else:
        return None
    
    
M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
if M is None:
    print("Error!")
(matches, H, status) = M
print(H)

# Apply panorama correction
width = img1.shape[1] + img2.shape[1]
height = img1.shape[0] + img2.shape[0]

result = cv2.warpPerspective(img1, H, (width, height))
result[0:img2.shape[0], 0:img2.shape[1]] = img2

plt.figure(figsize=(20,10))
plt.imshow(result)

plt.axis('off')
plt.show()


# transform the panorama image to grayscale and threshold it 
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

# Finds contours from the binary image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# get the maximum contour area
c = max(cnts, key=cv2.contourArea)

# get a bbox from the contour area
(x, y, w, h) = cv2.boundingRect(c)

# crop the image to the bbox coordinates
result = result[y:y + h, x:x + w]

# show the cropped image
plt.figure(figsize=(20,10))
plt.imshow(result)






