#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 12:00:02 2020

@author: pavankunchala
"""

import face_recognition

known_image_1  =face_recognition.load_image_file('Pavan-1.jpg')
known_image_2  =face_recognition.load_image_file('Pavan-2.png')
known_image_3  =face_recognition.load_image_file('pavan-3.jpg')
known_image_4  =face_recognition.load_image_file('rock.png')



names = ["Pavan-1.jpg", "Pavan-2.png", "pavan-3.jpg", "rock.png"]

unknown_image =face_recognition.load_image_file("pavan-4.jpg")

# Calculate the encodings for every of the images:
known_image_1_encoding = face_recognition.face_encodings(known_image_1)[0]
known_image_2_encoding = face_recognition.face_encodings(known_image_2)[0]
known_image_3_encoding = face_recognition.face_encodings(known_image_3)[0]
known_image_4_encoding = face_recognition.face_encodings(known_image_4)[0]
known_encodings = [known_image_1_encoding, known_image_2_encoding, known_image_3_encoding, known_image_4_encoding]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare the faces:
results = face_recognition.compare_faces(known_encodings, unknown_encoding)

# Print the results:
print(results)

