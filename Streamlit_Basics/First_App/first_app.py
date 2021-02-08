import numpy as np 
import streamlit as st
import pandas as pd
import time

#let's add a title to 
st.title('Basic App')

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [5, 3, 2, 4],
    'second column': [10, 20, 30, 40]
}))

st.write(" # The Chart is here")

#Draw a  chart
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['l', 'm', 'n'])

st.line_chart(chart_data)

#Draw a map
st.write(""" # The Map is here
         So are you""")

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

if st.checkbox('Map'):
    st.map(map_data)

#Check box part

if st.checkbox('Show Chartdata'):
    chart_data = pd.DataFrame(
       np.random.randn(40, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(chart_data)
    
    
left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Press me?')
if pressed:
    right_column.write("Woohoo!")

expander = st.beta_expander("FAQ")
expander.write("Here you could put in some really, really long explanations...")


#let's start shoowing progress bar
 #let's add a placeholder
 
latest_iteration = st.empty()

bar = st.progress(0)

for i in range(100):
      # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'
