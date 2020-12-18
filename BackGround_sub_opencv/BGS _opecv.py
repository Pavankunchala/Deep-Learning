import cv2
import numpy as np
import matplotlib.pyplot as plt

cap =cv2.VideoCapture("walk.png")

#creating a backGround subractor
backSubK= cv2.createBackgroundSubtractorKNN()

#for another background subractorMOG2
backSubM = cv2.createBackgroundSubtractorMOG2()

img = cv2.imread('walk.png')

#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


fgMask = backSubM.apply(img)



cv2.imshow('fgMask',fgMask)



