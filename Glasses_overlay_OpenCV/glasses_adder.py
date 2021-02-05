#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 22:24:57 2020

@author: pavankunchala
"""

import cv2
import numpy as np

# Load cascade classifiers for face and eyepair detection:
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyepair_cascade = cv2.CascadeClassifier("haarcascade_mcs_eyepair_big.xml")

# Load glasses image. The parameter -1 reads also de alpha channel (if exists)
# Open 'glasses.sgv' to see more glasses that can be used
# Therefore, the loaded image has four channels (Blue, Green, Red, Alpha):
img_glasses = cv2.imread('sun.png', -1)

#img_glasses = cv2.resize(img_glasses,(0,0), fx = 2 , fy = 2)


# Create the mask for the glasses:
img_glasses_mask = img_glasses[:, :, 3]
glass_mask = cv2.merge((img_glasses_mask,img_glasses_mask,img_glasses_mask))

glass_mask = np.uint8(glass_mask/255)

# cv2.imshow("img glasses mask", img_glasses_mask)

# Convert glasses image to BGR (eliminate alpha channel):
img_glasses = img_glasses[:, :, 0:3]

# You can use a test image to adjust the ROIS:
#test_face = cv2.imread("face_test.png")

# Create VideoCapture object to get images from the webcam:
video_capture = cv2.VideoCapture('test2.mov')

frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

size = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc(*"MP4V")
out = cv2.VideoWriter('output.mp4',fourcc, 25, size)
while True:
    # Capture frame from the VideoCapture object:
    ret, frame = video_capture.read()

    # Just for debugging purposes and to adjust the ROIS:
    # frame = test_face.copy()

    # Convert frame to grayscale:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the function 'detectMultiScale()'
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over each detected face:
    for (x, y, w, h) in faces:
        # Draw a rectangle to see the detected face (debugging purposes):
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Create the ROIS based on the size of the detected face:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect the eyepair inside the detected face:
        eyepairs = eyepair_cascade.detectMultiScale(roi_gray)

        # Iterate over the detected eyepairs (inside the face):
        for (ex, ey, ew, eh) in eyepairs:
            # Draw a rectangle to see the detected eyepair (debugging purposes):
            # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)

            # Calculate the coordinates where the glasses will be placed:
            x1 = int(ex - ew / 5)
            x2 = int((ex + ew) + ew / 5)
            y1 = int(ey)
            y2 = int(ey + eh + eh /2)

            if x1 < 0 or x2 < 0 or x2 > w or y2 > h:
                continue

            # Draw a rectangle to see where the glasses will be placed (debugging purposes):
            #cv2.rectangle(roi_color, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Calculate the width and height of the image with the glasses:
            img_glasses_res_width = int(x2 - x1) 
            img_glasses_res_height = int(y2 - y1 ) 

            # Resize the mask to be equal to the region were the glasses will be placed:
            mask = cv2.resize(img_glasses_mask, (img_glasses_res_width , img_glasses_res_height  ))

            # Create the invert of the mask:
            mask_inv = cv2.bitwise_not(mask)

            # Resize img_glasses to the desired (and previously calculated) size:
            img = cv2.resize(img_glasses, (img_glasses_res_width , img_glasses_res_height  ))

            # Take ROI from the BGR image:
            roi = roi_color[y1:y2, x1:x2]

            # Create ROI background and ROI foreground:
            roi_bakground = cv2.bitwise_and(roi, roi, mask=mask_inv)
            roi_foreground = cv2.bitwise_and(img, img, mask=mask)

            # Show both roi_bakground and roi_foreground (debugging purposes):
            # cv2.imshow('roi_bakground', roi_bakground)
            # cv2.imshow('roi_foreground', roi_foreground)

            # Add roi_bakground and roi_foreground to create the result:
            res = cv2.add(roi_bakground, roi_foreground)

            # Set res into the color ROI:
            roi_color[y1:y2, x1:x2] = res

            break

    # Display the resulting frame
    out.write(frame)
    #cv2.imshow('Snapchat-based OpenCV glasses filter', frame)

    # Press any key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything:
video_capture.release()
cv2.destroyAllWindows()