import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np 
import pandas as pd 
import dlib
from tqdm import tqdm
import cvlib as cv
import face_recognition



x1 = pd.read_csv("TEST.csv")
print(x1.head(5))
x2=[]
facedetector= dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

for i in tqdm(range(x1.shape[0])):
    z1 = "train_FD/image_data/" + x1["Name"][i]

    img = cv2.imread(z1,0)
    face = face_recognition.face_locations(img,2,'cnn')
    x2.append(len(face))

submissions = pd.DataFrame({'Name':x1['Name'],'HeadCount':x2})
submissions.to_csv("solution_analytic_vidhya_FD_2.csv",index = False)







