#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:11:33 2020

@author: pavankunchala
"""

import cv2

capture = cv2.VideoCapture("http://www.cnps.cat/moll_b_c_e.html")

frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

# Check if camera opened successfully
if capture.isOpened()is False:
    print("Error opening the camera")
    
frame_index = 0
 

while capture.isOpened() is True:
    ret, frame = capture.read()
    
    if ret is True:
       
        # Display the captured frame:
        cv2.imshow('Input frame from the camera', frame)
        
        b,g,r = cv2.split(frame)
        mis_frame = cv2.merge([g,r,b])
        

        # Convert the frame captured from the camera to grayscale:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        

        # Display the grayscale frame:
     
        cv2.imshow('Input frame from the camera', gray_frame)
        cv2.imshow("The discolured image of the input", mis_frame)
        
        
        #press c key to capture
        if cv2.waitKey(2000) & 0xFF  == ord('c'):
            
            frame_name = "camera_frame_{}.png".format(frame_index)
            gray_frame_name = "grayscale_camera_frame_{}.png".format(frame_index)
            cv2.imwrite(frame_name, frame)
            cv2.imwrite(gray_frame_name, gray_frame)
            frame_index += 1
            
        
      
 
        # Press q on keyboard to exit the program
        if cv2.waitKey(2000) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
    
    
# Release everything:
capture.release()
cv2.destroyAllWindows()
