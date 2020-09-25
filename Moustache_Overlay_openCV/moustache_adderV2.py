#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:49:44 2020

@author: pavankunchala
"""

import os,sys,time
import numpy as np
import cv2

def overlay_with_transparency(background,overlay,X,Y):
    bh,bw = background.shape[:2]
    
     # overlay dimensions
    oh,ow = overlay.shape[:2]
    
    # overlay start (top-left) is greater than background
    if X >=bw or  Y >= bh:
        return background
    
    # overlay end is less than background
    if ow + X <= 0 or oh + Y <= 0:
    
        return background
    
    #resize the overlay
    if ow + X <= 0 or oh + Y <= 0:
        return background
    
    if X + ow > bw:
        ow = bw - X
        overlay = overlay[:,:ow] # cut 2nd layer (cols) lengths
    if Y + oh > bh:
        oh = bh - Y
        overlay = overlay[:oh] # cut first layer (rows) length
        
    # no alpha channel (opaque)
    if overlay.shape[2] == 3:
        background[Y:Y+oh,X:X+ow] = overlay 
        
     # alpha channel (transparent)
    else:

        # make a mask of overlay alpha values as decimals of max value 255
        mask = overlay[...,3:]/255.0

        # the alpha channel formula: image = alpha*foreground + (1-alpha)*background
        background[Y:Y+oh,X:X+ow] = mask*overlay[...,:3] + (1-mask)*background[Y:Y+oh,X:X+ow]

    # done
    return background

camera = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml") # default cascade
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml") # default cascade
nose_cascade = cv2.CascadeClassifier(os.path.join(os.getcwd(),'haarcascade_mcs_nose.xml')) # local included cascade

mustache = cv2.imread('img.png',-1)
oh,ow = mustache.shape[:2]

draw_mustache = True  # toggle = m
draw_nose     = False # toggle = n
draw_box      = False # toggle = b
draw_eyes     = False # toggle = v

smooth_mustache = True
lastplace = (None,None)

while 1:

    # grab a frame from camera
    (grabbed,colorframe) = camera.read()
    
    # end of camera feed
    if not grabbed:
        break

    # make grayscale frame
    grayframe = cv2.cvtColor(colorframe,cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(
        grayframe,
        scaleFactor=1.1, # default 1.1
        minNeighbors=5, # default 3
        minSize=(50,50), # default (0,0)
        )
    for (x,y,w,h) in faces:

        # we need noses
        if draw_nose or draw_mustache:

            # only look for noses in the center 50% of the face
            w2 = w//2
            h2 = h//2
            x2 = x + w2//2
            y2 = y + h2//2
            grayroi = grayframe[y2:y2+h2,x2:x2+w2]

            # get noses
            noses = nose_cascade.detectMultiScale(grayroi,minSize=(w//6,h//6))
            for (x3,y3,w3,h3) in noses:

                # draw mustache
                if draw_mustache:
                    # scale mustache to 85% of face frame
                    scale = 0.85*w/ow
                    overlay = cv2.resize(mustache,(0,0),fx=scale,fy=scale)
                    oh2,ow2 = overlay.shape[:2]
                    # add to color frame where adjusted from middle-center of nose roi
                    x4 = x2+x3+(w3//2) - (ow2//2)
                    y4 = y2+y3+(y3//2) + (oh2//12)
                    if smooth_mustache and lastplace != (None,None):
                        x4 = (x4+lastplace[0])//2
                        y4 = (y4+lastplace[1])//2
                    colorframe = overlay_with_transparency(colorframe,overlay,x4,y4)
                    lastplace = (x4,y4)

                # draw nose box on top of mustache
                if draw_nose:
                    cv2.rectangle(colorframe,(x2+x3,y2+y3),(x2+x3+w3,y2+y3+h3),(0,255,000),1)

        # find and draw eye boxes
        if draw_eyes:
            grayroi = grayframe[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(grayroi,minSize=(w//10,h//10))
            for (x3,y3,w3,h3) in eyes[:2]:
                cv2.rectangle(colorframe,(x+x3,y+y3),(x+x3+w3,y+y3+h3),(255,0,0),1)

        # draw face box on top of everything
        if draw_box:
            cv2.rectangle(colorframe,(x,y),(x+w,y+h),(0,0,255),1)

    # display
    cv2.imshow("Face Detection: OpenCV Python3",colorframe)

    # key delay and action
    # these will toggle the draw flag
    # use "q" to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        draw_box = not draw_box
    elif key == ord('m'):
        draw_mustache = not draw_mustache
    elif key == ord('n'):
        draw_nose = not draw_nose
    elif key == ord('v'):
        draw_eyes = not draw_eyes
    elif key != 255:
        print('key:',[chr(key)])

# release camera
camera.release()

# close all windows
cv2.destroyAllWindows()








