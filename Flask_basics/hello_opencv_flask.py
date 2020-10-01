#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:23:44 2020

@author: pavankunchala
"""

import cv2
from flask import Flask,request,make_response
import numpy as np
import urllib.request

app = Flask(__name__)

@app.route('/canny', methods = ['GET'])

def  canny_processing():
    
    #get thhe img
    with urllib.request(request.args.get('url')) as url:
        image_array = np.asarray(bytearray(url.read()), dtype=np.uint8)
        
    img_opencv = cv2.imdecode(image_array,-1)
    
    gray = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2GRAY)
    
    #perform canny edge detection
    edges = cv2.Canny(gray,100,200)
    
    # Compress the image and store it in the memory buffer:
    retval, buffer = cv2.imencode('.jpg', edges)
    
      # Build the response:
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'

    # Return the response:
    return response


if __name__ == "__main__":
    # Add parameter host='0.0.0.0' to run on your machines IP address:
    app.run(host='0.0.0.0')
