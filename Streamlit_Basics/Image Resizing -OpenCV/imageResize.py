import streamlit as st
import cv2
from PIL import Image
import numpy as np

st.title('Image Resizing with OpenCV')

st.subheader('Upload the image')

uploaded_file = st.file_uploader('Choose a image file',)

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")
    #image  = uploaded_file.read()
    
    st.subheader('Define new Width and Height')
    
    
    width = int(st.number_input('Input a new a Width'))
    height = int(st.number_input('Input a new a Height'))
    img = cv2.imread('am.png')
    
    points = ((width, height))
    
    resized_image = cv2.resize(img , points, interpolation = cv2.INTER_LINEAR)
    
    st.image(resized_image[:,:,::-1])
    