#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 09:56:12 2020

@author: pavankunchala
"""

"""
Download the training data from the below link:

www.reddit.com/r/datasets/comments/3akhxy/the_ largest_midi_collection_on_the_internet/  

Download the following application for running the MIDI files
https:// musescore.org/en
  
"""


#importing the libraries

import tensorflow as tf
from music21 import converter, instrument,note,chord, stream
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from keras.layers.core import Dense , Activation, Flatten
from keras.layers import GRU,Convolution1D,Convolution2D, Flatten,Dropout, Dense
from keras import utils as np_utils
from tensorflow.keras import layers

print(os.getcwd())

music_dir = "//Users//pavankunchala//Downloads//PROJECTS//Deep-Learning//Music_Generator_GRU//music_files"

os.chdir(music_dir)


musical_note = []
offset = []
instrumentlist= []

music_dir2="//Users//pavankunchala//Downloads//PROJECTS//Deep-Learning//Music_Generator_GRU//music_files"
#loading the data

#taking 5 of the random files d
filenames = random.sample(os.listdir(music_dir), 5)

musiclist = os.listdir(music_dir)

#feature extraction

for file in filenames:
    
    matching = [s for s in musiclist if file.split('_')[0] in s]
    print("Matching:",matching)
    
    r1 = matching[random.randint(1,len(matching))]
    
    string_midi = converter.parse(r1)
    
    parsednotes = None
    
    parts = instrument.partitionByInstrument(string_midi)
    
    instrumentlist.append(parts.parts[0].getInstrument().instrumentname)
   
    #file has instrument parts
    if parts: 
        parsednotes = parts.parts[0].recurse()
        
    #file has flat notes
    else:
        parsednotes = string_midi.flat.notes
        
    #detect offsets
    for element in parsednotes:
        offset.append(element.offset)
        
        if isinstance(element, note.Note):
            musical_note.append(str(element.pitch))
            
        elif isinstance(element, chord.Chord):
            
            musical_note.append('.'.join(str(n) for n in element.normalOrder))
            
            
#Exploritaory data analysis
pd.Series(instrumentlist).value_counts()
#pd.Series(instrumentlist).value_counts()

pd.Series(musical_note).value_counts()
#pd.Series(musical_note).value_counts()

offset = [float(item) for item in offset]
plt.plot(offset)

plt.show()
        

#Data prep


sequence_length = 100

# Arranging notes and chords in ascending order
pitchcategory = sorted(set(item for item in musical_note))

#one hot encoding

note_encoding = dict((note, number) for number, note in enumerate(pitchcategory))

model_input_original = []

model_output = []

# Prepare input and output data for model
for i in range(0, len(musical_note) - sequence_length, 1):
    
    sequence_in = musical_note[i:i + sequence_length]
    
    sequence_out = musical_note[i + sequence_length]
    
model_input_original.append([note_encoding[char] for char in sequence_in])

model_output.append(note_encoding[sequence_out])

n_patterns = len(model_input_original)

model_input = np.reshape(model_input_original, (n_patterns, sequence_length, 1))

model_output = np_utils.to_categorical(model_output)

Len_Notes = model_output.shape[1]

model_input = model_input / float(Len_Notes)


#strycture the model
model_GRU = tf.keras.models.Sequential()

model_GRU.add(layers.GRU(16,input_shape=(model_input. shape[1], model_input.shape[2]),return_sequences=True))  

model_GRU.add(layers.Dropout(0.3))

model_GRU.add(layers.GRU(64, return_sequences=True))

model_GRU.add(layers.Dropout(0.3))

model_GRU.add(layers.GRU(64))

model_GRU.add(layers.Dense(16))

model_GRU.add(layers.Dropout(0.3))

model_GRU.add(layers.Dense(Len_Notes))

model_GRU.add(layers.Activation('softmax'))

model_GRU.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model_GRU.summary()

#training
int_to_note = dict((number, note) for number, note in enumerate(pitchcategory))

pattern = model_input_original[0]

prediction_output = []

model_GRU.fit(model_input, model_output, epochs=30, batch_size=64)

for note_index in range(500):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    
    prediction_input = prediction_input / float(Len_Notes)
    
    prediction_GRU = model_GRU.predict(prediction_input, verbose=0)
    
    index_GRU = np.argmax(prediction_GRU)
    
    index = index_GRU
    
    result = int_to_note[index]
    
    prediction_output.append(result)
    
    pattern = np.append(pattern,index)
    
    pattern = pattern[1:len(pattern)]
    
    
offlen = len(offset)

DifferentialOffset = (max(offset)-min(offset))/ len(offset)

offset2 = offset.copy()

output_notes = []
i= 0
offset = []
initial = 0

for i in range(len(offset2)):
    offset.append(initial)
    initial = initial+DifferentialOffset
    
i=0
for pattern in prediction_output:
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        
        for check_note in notes_in_chord:
            gen_note = note.Note(int(check_note)) 
            gen_note.storedInstrument = instrument.Guitar()
    



        
    
    
    
    
    
    



