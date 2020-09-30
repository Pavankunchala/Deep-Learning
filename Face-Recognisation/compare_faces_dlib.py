#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:15:27 2020

@author: pavankunchala
"""

import cv2
import dlib
import numpy as np

pose_predictor_5_point = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()



"""Returns the ordered distances and names when comparing a list of face encodings against a candidate 
 to check"""
def compare_faces_ordered(encodings,face_names,encoding_to_check):
    
    
 
    distances = list(np.linalg.norm(encodings- encoding_to_check,axis = 1))
    
    return zip(*sorted(zip(distances,face_names)))


def compare_faces(encodings,encoding_to_check):
    
    return list(np.linalg.norm(encodings- encoding_to_check,axis = 1))

def face_encodings(face_image,number_of_times_to_upsample = 1,num_jitters =1):
    
    #detect faces
    face_locations = detector(face_image,number_of_times_to_upsample)
    
    #detect landmarks
    raw_landmarks = [pose_predictor_5_point(face_image,face_location) for face_location in face_locations]
    
    # Calculate the face encoding for every detected face using the detected landmarks for each one
    
    return  [np.array(face_encoder.compute_face_descriptor(face_image,raw_landmark_set , num_jitters)) for 
             raw_landmark_set in raw_landmarks]

# Load images:
known_image_1 = cv2.imread("Pavan-1.jpg")
known_image_2 = cv2.imread("Pavan-2.png")
known_image_3 = cv2.imread("pavan-3.jpg")
known_image_4 = cv2.imread("rock.png")
unknown_image = cv2.imread("pavan-4.jpg")

# Convert image from BGR (OpenCV format) to RGB (dlib format):
known_image_1 = known_image_1[:, :, ::-1]
known_image_2 = known_image_2[:, :, ::-1]
known_image_3 = known_image_3[:, :, ::-1]
known_image_4 = known_image_4[:, :, ::-1]
unknown_image = unknown_image[:, :, ::-1]

# Crate names for each loaded image:
names = ["Pavan-1.jpg", "Pavan-2.png", "pavan-3.jpg", "rock.png"]


# Create the encodings:
known_image_1_encoding = face_encodings(known_image_1)[0]
known_image_2_encoding = face_encodings(known_image_2)[0]
known_image_3_encoding = face_encodings(known_image_3)[0]
known_image_4_encoding = face_encodings(known_image_4)[0]
known_encodings = [known_image_1_encoding, known_image_2_encoding, known_image_3_encoding, known_image_4_encoding]
unknown_encoding = face_encodings(unknown_image)[0]

# Compare faces:
computed_distances = compare_faces(known_encodings, unknown_encoding)
computed_distances_ordered, ordered_names = compare_faces_ordered(known_encodings, names, unknown_encoding)


# Print obtained results:
print(computed_distances)
print(computed_distances_ordered)
print(ordered_names)
    
    
    
     
      
      