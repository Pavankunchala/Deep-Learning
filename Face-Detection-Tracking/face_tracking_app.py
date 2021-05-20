from mtcnn.mtcnn import MTCNN 
import cv2
from sort import *
from util import *
import util
import streamlit as st
import numpy as np




def main():
    
#title
    st.title('Face Tracking Application')

#sidebar title
    st.sidebar.title('Face Tracking Application')

#markdown
    st.markdown('In this application we will track the face of a person by their Face ID')

    st.sidebar.markdown('In this application we will track the face of a person by their Face ID')

#min confidence 
    #onfidence = st.sidebar.slider('Min Confidence',min_value=0.0, max_value=1.0, value = 0.4)
    min_face_size = st.sidebar.slider('Min Face Size you want to detect',min_value=1,max_value=200,value=40)
    util.required_height = st.sidebar.slider('Height of the Face',min_value=0, max_value=200,value = 10)
    
    

#define the input file
    input_file = st.text_input('Input File Path',value = 'test.mp4')

    filename  = open(input_file,'rb')

    video_bytes = filename.read()
    


    st.sidebar.text('Input Video')
    st.sidebar.video(video_bytes)




    face_detector = MTCNN(min_face_size = min_face_size)  #Initializing MTCNN detector object
    face_tracker  = Sort(max_age=50)
    stframe = st.empty()

    vid = cv2.VideoCapture(input_file)
    video_frame_cnt = int(vid.get(7))
    video_width = int(vid.get(3))
    video_height = int(vid.get(4))
    video_fps = int(vid.get(5))
    record_video = True
    out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), video_fps, (video_width, video_height))
    
    output = st.checkbox('Save The Video',value =False)

    
    
    while True:
    
        ret , frame = vid.read()
    
   
        original_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_detector.detect_faces(frame)
        box = []
        for i in range(len(result)):
            
        
            box_ = result[i]["box"]
            print("Face Detectes: Box=",box_)
            box.append([box_[0],box_[1],box_[0]+box_[2],box_[1]+box_[3],result[i]["confidence"] ])
        
        dets = np.array(box)
        track_bbx,pts = face_tracker.update(dets)
    
        _MODEL_SIZE = [original_frame.shape[1], original_frame.shape[0]]
        
        original_frame = track_img(original_frame, track_bbx, _MODEL_SIZE, pts)
    
        #if record_video:
            
            #out.write(original_frame)
            
        
        #cv2.imshow('Frame',original_frame)
        stframe.image(original_frame,use_column_width=True,channels = 'BGR')
            
        key = cv2.waitKey(1) & 0xFF
        
        out.write(original_frame)            
        if output:
            
            
            filename1 = open('output.mp4','rb')
            video_bites1 = filename1.read()
            st.text("Output Video:")
            st.video(video_bites1)
            
            break

    



   

    
    
    
if __name__ == "__main__":
    main()

    
            
    

        
    
    
        
    
    


