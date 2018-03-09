# app.py

import logging
import random
import time

from flask import Flask, jsonify, request
import numpy as np
from scipy.misc import imread, imresize
import keras

from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

import cv2                
import matplotlib.pyplot as plt    

app = Flask(__name__)
app.config.from_object(__name__)

# Loading the face detector to detect faces
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')




@app.route('/')
def classify():

    file_path = request.args['file_path']
    app.logger.info("Classifying image %s" % (file_path),)

    # Load in an image to classify and preprocess it
    img = cv2.imread(human_files[3])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    app.logger.info("Number of faces detected: %d", len(faces))
    

    return jsonify(predictions.tolist())

if __name__ == '__main__':

    app.run(debug=True, port=8009)