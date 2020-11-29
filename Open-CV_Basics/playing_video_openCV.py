import cv2
import numpy as np

cap = cv2.VideoCapture("run1.mp4")

while cap.isOpened():
    ret,frame = cap.read()
    
    #if Frame is read correctly ret is  True
    if not ret:
        print('Cant find the file  check it once again..')
        
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    gray = cv2.resize(gray,(0,0),fx = 0.5,fy = 0.5)
    
    cv2.imshow("Gray frame",gray)
    
    if cv2.waitKey(1) == ord('q'):
        
        break
    

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()