#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:53:37 2020

@author: pavankunchala

"""

"""
Download the dataset from here
https://drive.google.com/open?id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM
"""



#import libraries

import tensorflow as tf
import numpy as np
import pandas as pd
import re
import keras
from keras import Model
from tensorflow.keras.layers import Dense,LSTM,Flatten, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from keras_preprocessing.text import Tokenizer
from keras.initializers import glorot_uniform
from sklearn import model_selection

#loading the DATA

with open('train.csv', 'r') as file:
    
    text = file.readlines()
    
#creating an empty dataframe
x_train = pd.DataFrame()

#fill in the dataframe
word = []
label = []
for n in text:
    n = n.split()
    label.append(1) if n[0] =="__label__2" else label.append(0)
    word.append(" ".join(n[1:]))
    
x_train['consumer_review'] = word
x_train['polarity_label'] = label

#view the dataframe
#print("X Train =",x_train)

#using only 20 percent data
_ , x_set , _,y_set = model_selection.train_test_split(x_train['consumer_review'],
                                                       x_train['polarity_label'],
                                                       test_size = 0.02)

" Cleaning the data"

# data cleaning func

def data_prep(in_tex):
    #remove punctuations and numbers
    out_tex = re.sub('[^a-zA-Z]', ' ', in_tex)
    # Convert upper case to lower case
    out_tex="".join(list(map(lambda x:x.lower(),out_tex)))
    # Remove single character
    out_tex= re.sub(r"\s+[a-zA-Z]\s+", ' ', out_tex)
    
    return out_tex

#crearing a new list with cleaN dats

text_set = []

for reviews in list(x_set):
    text_set.append(data_prep(reviews))
    
# creating a new dataframw to storw the data
x_train = pd.DataFrame()

x_train['consumer_review'] = text_set
x_train['polarity_label'] = list(y_set)

#split data into 70% train and 30% test
x_train, x_test ,y_train,y_test = model_selection.train_test_split(x_train['consumer_review'],
                                                       x_train['polarity_label'],
                                                       test_size = 0.3)

#converting to an array
x_train = (x_train.values.tolist())

x_test = (x_test.values.tolist())

y_train = (y_train.values.tolist())

y_test = (y_test.values.tolist())
    

#tokenizer

tokenizer  = Tokenizer()

tokenizer.fit_on_texts(x_train)

word_index= tokenizer.word_index

total_size = len(word_index)+1

print(total_size)


#texts to sequence
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

#add padding to ensure the same lenght

max_length = 100

x_train =  pad_sequences(x_train , padding='post', maxlen = max_length)
x_test =  pad_sequences(x_test , padding='post', maxlen = max_length)

#creating the model 
 
model = Sequential()
#Adding a Emnediing layer
model.add(Embedding(total_size,20,input_length=max_length))

#adding a LSTM layer
model.add(LSTM(32,dropout=0.2,recurrent_dropout=0.2))

#addinf a Dense layer
model.add(Dense(1,activation='sigmoid'))

#compiling rhe model
model.compile(optimizer='adam',loss = 'binary_crossentropy', metrics=['acc'])

print(model.summary())

#fitting the model

model.fit(x_train,y_train, batch_size=128,epochs=5, verbose=1, validation_data=(x_test, y_test))

model.save("model.h5")



    
    
