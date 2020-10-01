#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:07:38 2020

@author: pavankunchala
"""

from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


if __name__ == "__main__":
    # Add parameter host='0.0.0.0' to run on your machines IP address:
    app.run(host='0.0.0.0')