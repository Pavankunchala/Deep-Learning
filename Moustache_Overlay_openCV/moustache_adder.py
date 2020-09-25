#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:22:45 2020

@author: pavankunchala
"""

import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")

#load the imagew
img_moustache = cv2.imread("it.png")

#create a mask fot moustache
img_moustache_mask  = img_moustache[:,:,2]
#cv2.imshow("img moustache mask", img_moustache_mask)

test_face = cv2.imread("face.png")

img_moustache = img_moustache[:, :, 0:3]

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame from the VideoCapture object:
    ret, frame = video_capture.read()
    #frame = cv2.imread('face.jpg')
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #detect faces using multi scale
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    
    #iterate over each  detected face
    for(x , y, w, h) in faces:
        
        #drawinf a rectangle
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        
        
        #Create the ROIS based on the size of the detected face:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
        #detect nose
        noses = nose_cascade.detectMultiScale(roi_gray)
        
        for (nx,ny,nw,nh) in noses:
            # Draw a rectangle to see the detected nose (debugging purposes):
           # cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 255), 2)
            
            #cordinatew where moustache will be places
            x1 = int(nx - nw / 2)
            x2 = int(nx + nw / 2 + nw)
            y1 = int(ny + nh / 2 + nh / 8)
            y2 = int(ny + nh + nh / 4 + nh / 6)
            
            if x1 < 0 or x2 < 0 or x2 > w or y2 > h:
                continue
            
            #cv2.rectangle(roi_color, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            #calcualte the height width off moustahce
            img_moustache_res_width = int(x2 - x1)
            img_moustache_res_height = int(y2 - y1)
            
             # Resize the mask to be equal to the region were the glasses will be placed:
            mask = cv2.resize(img_moustache_mask, (img_moustache_res_width, img_moustache_res_height))
            
            # Create the invert of the mask:
            mask_inv = cv2.bitwise_not(mask)

            # Resize img_glasses to the desired (and previously calculated) size:
            img = cv2.resize(img_moustache, (img_moustache_res_width, img_moustache_res_height))

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
    # Display the resulting frame:
    cv2.imshow('Snapchat-based OpenCV moustache overlay', frame)

    # Press any key to exit:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything:
video_capture.release()
cv2.destroyAllWindows()




        
        
        
