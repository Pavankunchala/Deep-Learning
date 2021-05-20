'''
Author : Santosh Adhikari 
        2020 - June-13 
Real Time Face Detection and Tracking Using MTCNN and SORT 

'''

from mtcnn.mtcnn import MTCNN 
import cv2
from sort import *
from util import *


#vid = cv2.VideoCapture(0)
vid = cv2.VideoCapture('test.mp4')
video_frame_cnt = int(vid.get(7))
video_width = int(vid.get(3))
video_height = int(vid.get(4))
video_fps = int(vid.get(5))
record_video = True
#ecord_video = False
if record_video:
    out = cv2.VideoWriter('data/outvideo.avi',cv2.VideoWriter_fourcc('M','J','P','G'), video_fps, (video_width, video_height)) # for writing Video
face_detector = MTCNN()  #Initializing MTCNN detector object
face_tracker  = Sort(max_age=50)   #Initializing SORT tracker object

ret , frame = vid.read()
while ret:
    try: 
        ret , frame = vid.read()
        original_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_detector.detect_faces(frame)
        min_confidence= 0.4

        box = []
        for i in range(len(result)):
            
            box_ = result[i]["box"]
            print("Face Detectes: Box=",box_)
            box.append([box_[0],box_[1],box_[0]+box_[2],box_[1]+box_[3],result[i]["confidence"] ])
            
            
            

        
        dets = np.array(box)
        track_bbx,pts = face_tracker.update(dets)
        

        _MODEL_SIZE = [original_frame.shape[1], original_frame.shape[0]]
        
        original_frame = track_img(original_frame, track_bbx, _MODEL_SIZE, pts)
        
        

        if record_video:
            out.write(original_frame)

        cv2.imshow("out_frame", original_frame)
    except Exception as e:
        print (getattr(e, 'message', repr(e)))

    keyPressed = cv2.waitKey(1) & 0xFF
    if keyPressed == 27: # ESC key
        break

# When everything is done, release the capture
vid.release()
cv2.destroyAllWindows()
