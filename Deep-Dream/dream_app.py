import streamlit as st

import streamlit.components.v1 as stc
from PIL import Image
from tensorflow.keras.preprocessing.image import *
from deepDream import DeepDreamer
import numpy as np

DEMO_IMAGE = 'new_img.jpg'

HTML_BANNER = """
<div style="background-color:Orange;padding:10px;border-radius:10px">
<h1 style="color:Black;text-align:center;">Deep Dreamer</h1>
</div>
"""


stc.html(HTML_BANNER)

img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg"])

if img_file_buffer is not None:

    image = np.array(Image.open(img_file_buffer))

else:

    demo_image = DEMO_IMAGE

    image = np.array(Image.open(demo_image))

original_image = image

st.sidebar.subheader('Original Image')

st.sidebar.image(image,use_column_width = True)

def load_image(image):
    #image = load_img(image_path)
    image = img_to_array(image)

    return image

deep= DeepDreamer()

dreamy_image = deep.dream(image = original_image)


st.subheader('Dreamy Image')

st.image(dreamy_image,use_column_width= True)



