import cv2
import matplotlib.pyplot as plt

import numpy as np

video_capture = cv2.VideoCapture("street.mp4")


frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

size = (frame_width, frame_height)
fps = video_capture.get(cv2.CAP_PROP_FPS)



fourcc = cv2.VideoWriter_fourcc(*"MP4V")
out = cv2.VideoWriter('face-output1.mp4',fourcc, fps, size)

#net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")

while True:
    ret, frame = video_capture.read()
    
    #frame = cv2.resize(frame,(0,0),fx = 0.5, fy = 0.5)
    
    (h, w) = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104., 117., 123.], False, False)
    
    net.setInput(blob)
    detections = net.forward()
    
    detected_faces = 0
    
    for i in range(0, detections.shape[2]):
        
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.2:
            detected_faces += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.3f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 3)
            #cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    out.write(frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break


video_capture.release()
cv2.destroyAllWindows()
