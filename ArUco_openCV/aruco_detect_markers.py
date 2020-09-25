#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:15:35 2020

@author: pavankunchala
"""

#we are gng to detect the markers using WebCam

import cv2

aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)

#create the parameters object
parameters = cv2.aruco.DetectorParameters_create()

#capture the video 
capture = cv2.VideoCapture(0)

while True:
    
    #capture frame by frame
    ret , frame= capture.read()
    
    #converting the frame to gray scale
    gray_frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY)
    
    # We call the function 'cv2.aruco.detectMarkers()'
    # The first parameter is the image where the markers are going to be detected
    # The second parameter is the dictionary object
    # The third parameter establishes all the parameters that can be customized during the detection process
    
    corners, ids, rejected_corners = cv2.aruco.detectMarkers(gray_frame, aruco_dictionary, parameters=parameters)
    
    #draw detected markers
    frame = cv2.aruco.drawDetectedMarkers( image = frame, corners = corners, ids=ids, borderColor=(0, 255, 0))
    
    #drawinf rejected markers
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejected_corners, borderColor=(0, 0, 255))
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything:
capture.release()
cv2.destroyAllWindows()