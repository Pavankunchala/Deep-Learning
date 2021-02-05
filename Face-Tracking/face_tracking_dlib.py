#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:06:54 2020

@author: pavankunchala
"""

import cv2
import dlib

def draw_text_info():
    """Draw text information"""

    # We set the position to be used for drawing text and the menu info:
    menu_pos_1 = (10, 20)
    menu_pos_2 = (10, 40)

    # Write text:
    cv2.putText(frame, "Use '1' to re-initialize tracking", menu_pos_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    if tracking_face:
        cv2.putText(frame, "tracking the face", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    else:
        cv2.putText(frame, "detecting a face to initialize tracking...", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255))
    
capture = cv2.VideoCapture('test.mov')


frame_width = int(capture.get(3))
frame_height = int(capture.get(4))

size = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc(*"MP4V")
out = cv2.VideoWriter('output.mp4',fourcc, 25, size)

#frontal face detector
detector = dlib.get_frontal_face_detector()

#correlation tracker
tracker = dlib.correlation_tracker()

#if we are currently tracking 
tracking_face = False

while True:
    
    ret,frame= capture.read()
    
    #draw basic info
    draw_text_info()
    
    if tracking_face is False:

        
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #try to detect the face
        rects = detector(gray,0)
        
        # if we have detected a face
        if len(rects)>0: 
            
            #start tracking
            tracker.start_track(frame,rects[0])
            
            tracking_face = True
            
    if tracking_face is True:

        
        print(tracker.update(frame))
        # Get the position of the tracked object:
        pos = tracker.get_position()
        # Draw the position:
        cv2.rectangle(frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0, 255, 0), 3)

    # We capture the keyboard event
    key = 0xFF & cv2.waitKey(1)

    # Press '1' to re-initialie tracking (it will detect the face again):
    #if key == ord("1"):
     #   tracking_face = False

    # To exit, press 'q':
    #if key == ord('q'):
     #   break

    # Show the resulting image:
    out.write(frame)
    #cv2.imshow("Face tracking using dlib frontal face detector and correlation filters for tracking", frame)

# Release everything:
capture.release()
cv2.destroyAllWindows()
            
        
        

