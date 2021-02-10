import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import time, datetime

st.title('Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)

st.subheader('Area chart')
chart_data = pd.DataFrame( np.random.randn(20, 3), columns=['a', 'b', 'c'])

if st.checkbox('Show the Area chart'):
    st.area_chart(chart_data)
    
image =Image.open('am.png')

st.subheader('Just trying out few images')

st.image(image, caption='Sunrise by the mountains',use_column_width= True)


#let's add an video
st.subheader('Try out the video part')
video_file = open('it.mp4', 'rb')
video_bytes = video_file.read()


st.video(video_bytes)
    
#let's add few buttons Too
st.subheader('Button Part')

if st.button('How you doin?'):
    st.write("I am doin good how about you?")
    
else:
    st.write("Neva Mind")
   
   
#Radio 
st.subheader('Radio part here')

genre = st.radio("What's your favorite movie genre", ('Comedy', 'Drama', 'Documentary'))

if genre == 'Comedy':
    st.write("You have selected Comedy")
    
else:
    st.write("You shall not pass!")
    
#select box demonstrates
st.subheader('Selectbox demo')

option = st.selectbox('How would like your model to be deployed',('Flask','StreamLit','GCP','AWS'))

st.write('You selected:', option)

#multiselect demo 
st.subheader('Multiselect Demo')

options = st.multiselect(
     'What are your favorite colors',
     ['Green', 'Yellow', 'Red', 'Blue'],
     ['Yellow', 'Red'])

    
st.write('You selected:', options)


#slider example 
st.subheader('Slider Demo')

appointment = st.slider('Schedule your appointment:',value=(time(11, 30), time(12, 45)))

st.write("You're scheduled for:", appointment)


start_color, end_color = st.select_slider('Select a range of color wavelength',
             options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
             value=('red', 'blue'))

st.write('You selected wavelengths between', start_color, 'and', end_color)

#Text Input Demo

st.subheader('Text Input Demo')
title = st.text_input('Movie title', 'Life of Brian')
st.write('The current movie title is', title)



#Number input demo
st.subheader('Number Input Demo')

number = st.number_input('Insert a number')
st.write('The current number is ', number)

#Text area Demo
st.subheader('Text Area Demo')


txt = st.text_area('Text to analyze', '''
...     It was the best of times, it was the worst of times, it was
...     the age of wisdom, it was the age of foolishness, it was
...     the epoch of belief, it was the epoch of incredulity, it
...     was the season of Light, it was the season of Darkness, it
...     was the spring of hope, it was the winter of despair, (...)
...     ''')

#st.write('Sentiment:', run_sentiment_analysis(txt))







