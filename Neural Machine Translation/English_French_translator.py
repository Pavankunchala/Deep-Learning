#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 09:41:09 2020

@author: pavankunchala
"""

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
import matplotlib.ticker as ticker
import re
import io
import time
import pandas as pd



print(os.getcwd())
path = "/Users/pavankunchala/Downloads/PROJECTS/Deep-Learning/Neural Machine Translation"


path_to_file = os.path.dirname(path) + "/fra.txt"


# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
                 
      if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())
  
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
  w = w.strip()
  w = '<start> ' + w + ' <end>'
  return w

en_sentence = u"May I borrow this book?"
fren_sentence = u"PUIS-JE EMPRUNTER CE LIVRE?"

print(preprocess_sentence(en_sentence))
print(preprocess_sentence(fren_sentence).encode('utf-8'))

#removing the accents

def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
    return zip(*word_pairs)

en, sp = create_dataset(path_to_file, None)
print(en[-1])
print(sp[-1])







