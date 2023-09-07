import base64
from typing import Type
from urllib import response
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify, send_file
import os
import time

from io import BytesIO

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import mahotas as mh
import tensorflow as tf
import cv2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = load_model('./model/stroke_classification_model.h5')

@app.route('/')
@cross_origin()
def home():
    return "home"

@app.route('/ct-scan/add', methods=['POST'])
@cross_origin()
def addCTScan():
    data = request.json

    image64 = data['image']
    nik = data['nik']
    ts = time.time()

    img_width, img_height = 150, 150

    target_size = (img_width, img_height)
    img = load_image_from_base64(image64, target_size)
    name_file = "ct-scan/" + nik + "_" + str(ts) + '.jpg'
    img.save(name_file)
    return jsonify(name_file)


@app.route('/ct-scan', methods=['GET'])
@cross_origin()
def getCTScan():
    nik = request.args.get("nik")
    data_array = []
    dir_ct_scan = os.listdir("ct-scan")

    for ct_scan in dir_ct_scan:
        file_name = ct_scan.split("_")
        if file_name[0] == nik:
            time = file_name[1].split('.')
            temp = {"nik": file_name[0], "time": str(time[0])+ "." + str(time[1])}
            data_array.append(temp)

    return jsonify(data_array)

@app.route('/ct-scan/download', methods=['GET'])
@cross_origin()
def download_file():
    nik = request.args.get("nik")
    time = request.args.get("time")

    dir_ct_scan = "ct-scan/"
    file_path = dir_ct_scan + str(nik) + "_" + str(time) + '.jpg'

    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found."

@app.route('/schedule', methods=['GET'])
@cross_origin()
def getSchedule():
    return jsonify(
        {
        "patient_name": "John Doe",
        "doctor_name": "Dr. Smith",
        "appointment_date": "2023-09-15",
        "appointment_time": "10:30 AM",
        "procedure_type": "CT Scan",
        "clinic_location": "123 Main Street, Cityville",
        "contact_phone": "555-123-4567",
        "additional_notes": "Please arrive 15 minutes early for paperwork."
        }
    )

@app.route('/prediction', methods=['POST'])
@cross_origin()
def prediction():
    data = request.json

    image64 = data['image']

    img_width, img_height = 150, 150

    target_size = (img_width, img_height)
    img = load_image_from_base64(image64, target_size)

    gray_image = image_to_grayscale(img)

    blur_image = image_to_blur(gray_image)

    threshold_image = image_to_threshold(blur_image)

    morphological_image = image_to_morphological(threshold_image)

    feature = calculate_glcm_features(morphological_image)

    feature_standart = standarization(feature)

    feature_standart_flatten = feature_standart.values.flatten()

    feature_standart_array = np.array([feature_standart_flatten])

    res_predict = model.predict(feature_standart_array)

    res_max_index = np.argmax(res_predict)

    res_class = ""

    if (res_max_index == 0):
        res_class = "NOT STROKE"
    elif (res_max_index == 1):
        res_class = "STROKE HEMORRHAGIC"
    elif (res_max_index == 2):
        res_class = "STROKE ISCHEMIC"
    else:
        res_class = "E PREDICT"

    return jsonify(res_class)

def load_image_from_base64(base64_string, target_size):
    encoded_data = base64_string.split(',')[1]
    image_bytes = base64.b64decode(encoded_data)

    img = image.load_img(BytesIO(image_bytes), target_size=target_size)

    return img

def image_to_grayscale(img):
    gray_image = img.convert("L")
    return gray_image

def image_to_blur(img, kernel_size=(5, 5), sigma=0):
    image = np.array(img)
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image

def image_to_threshold(img, block_size=11):
    image = np.array(img)
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded_image

def image_to_morphological(img, kernel_size=3):
    image = np.array(img)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened_image

def calculate_glcm_features(img):
    image = np.array(img)

    glcm = mh.features.haralick(image.astype(np.uint8), ignore_zeros=True)

    contrast = np.mean(glcm[:, 2])
    dissimilarity = np.mean(glcm[:, 4])
    homogeneity = np.mean(glcm[:, 8])
    correlation = np.mean(glcm[:, 9])
    angular = np.mean(glcm[:, 1])
    energy = np.mean(glcm[:, 0])

    return [contrast, dissimilarity, homogeneity, correlation, angular, energy]

def standarization(feature):
    df_feature = pd.DataFrame(feature)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_feature)
    scaled_df = pd.DataFrame(scaled_data, columns=df_feature.columns)
    return scaled_df

if __name__ == '__main__':
    app.run(debug=True)