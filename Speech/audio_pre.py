#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:28:47 2020

@author: pavankunchala
"""

#importing the stuff


import librosa,librosa.display
import matplotlib.pyplot as plt
import numpy as np



file = 'wa.mp3'

#loading the audio file

signal , sr = librosa.load(file, sr = 22050)

#let's plot the waveplot
librosa.display.waveplot(signal, sr = sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


#frequency
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0,sr,len(magnitude))

#divinding the frequency
left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]

#We get this as symmetrical
plt.plot(frequency,magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

#plotting only left frequency

plt.plot(left_frequency,left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

#spectrogram

hop_length = 512 # in num. of samples
n_fft = 2048 # window in num. of samples

stft = librosa.core.stft(signal,hop_length=hop_length,n_fft = n_fft)
spectogram = np.abs(stft)

log_spectogram = librosa.amplitude_to_db(spectogram)

#plotting
librosa.display.specshow(log_spectogram,sr = sr,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()


#MFCCs
MFFcs = librosa.feature.mfcc(signal,n_fft = n_fft ,hop_length = hop_length, n_mfcc=15)
#plot again
librosa.display.specshow(MFFcs,sr = sr,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()




