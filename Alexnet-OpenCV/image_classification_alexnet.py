#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 22:29:44 2020

@author: pavankunchala
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Load the names of the classes:
rows = open('synset_words.txt').read().strip().split('\n')
classes = [r[r.find(' ') + 1:].split(',')[0] for r in rows]


net = cv2.dnn.readNetFromCaffe("bvlc_alexnet.prototxt", "bvlc_alexnet.caffemodel")

# Load input image:
image = cv2.imread("dog.jpg")


blob = cv2.dnn.blobFromImage(image, 1, (227, 227), (104, 117, 123))
print(blob.shape)

# Feed the input blob to the network, perform inference and get the output:
net.setInput(blob)
preds = net.forward()

# Get inference time:
t, _ = net.getPerfProfile()
print('Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()))

# Get the 10 indexes with the highest probability (in descending order)

indexes = np.argsort(preds[0])[::-1][:10]

# We draw on the image the class and probability associated with the top prediction:
text = "label: {}\nprobability: {:.2f}%".format(classes[indexes[0]], preds[0][indexes[0]] * 100)
y0, dy = 30, 30
for i, line in enumerate(text.split('\n')):
    y = y0 + i * dy
    cv2.putText(image, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 255),2)

# Print top 10 prediction:
for (index, idx) in enumerate(indexes):
    print("{}. label: {}, probability: {:.10}".format(index + 1, classes[idx], preds[0][idx]))

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(20, 12))
plt.suptitle("Image classification with OpenCV using AlexNet and caffe pre-trained models", fontsize=14,
             fontweight='bold')
fig.patch.set_facecolor('silver')

# Show the output image:
show_img_with_matplotlib(image, "AlexNet and caffe pre-trained models", 1)

# Show the Figure:
plt.show()