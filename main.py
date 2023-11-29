import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pytesseract as pt
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet
import easyocr as eo

from glob import glob
from skimage import io
from shutil import copy
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


#eo.download_and_install_model('craft', 'craft_mlt_25k.pth')
# settings
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# LOAD THE IMAGE
img = io.imread('TEST/TEST.jpeg')

#fig = px.imshow(img)
#fig.update_layout(width=700, height=400, margin=dict(l=10, r=10, b=10, t=10))
#fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
#fig.show()

# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('./Model3/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Chọn ngôn ngữ (ví dụ: tiếng Anh và tiếng Việt)
#languages = ['en', 'vi']

# Cấu hình tham số cho detector và recognizer
detector_params = {'name': 'craft', 'weights': 'craft_mlt_25k.pth'}
recognizer_params = {'name': 'latin'}

cropped_path = ""

# Tạo đối tượng Reader
reader = eo.Reader(
    lang_list=['en'],
    detector=detector_params,
    recognizer=recognizer_params
)

# extrating text
def extract_text(image,bbox):
    x,y,w,h = bbox
    roi = image[y:y+h, x:x+w]
    io.imsave(cropped_path, roi)
    #fig = px.imshow(roi)
    #fig.show()
    text = reader.readtext(roi)
    res = ""
    for dect in text:
        res = res + dect[1]
    return res
    # text = pt.image_to_string(roi);
    # text.strip()
    # return text
    # if 0 in roi.shape:
    #    return 'no number'
    # else:
    #    text = pt.image_to_string(roi)
    #    text = text.strip()
    #    return text



def get_detections(img,net):
    # 1.CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # 2. GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections

def non_maximum_supression(input_image,detections):

    # 3. FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE

    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    # 4.1 CLEAN
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    # 4.2 NMS
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)

    return boxes_np, confidences_np, index

def drawings(image,boxes_np,confidences_np,index):
    # 5. Drawings
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)

        license_text = extract_text(image,boxes_np[ind])
        #print("Plate is:" + license_text)

        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
        cv2.rectangle(image,(x,y+h),(x+w,y+h+25),(0,0,0),-1)

        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
        cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)

    return image

# predictions flow with return result
def yolo_predictions(img,net):
    # step-1: detections
    input_image, detections = get_detections(img,net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    # step-3: Drawings
    result_img = drawings(img,boxes_np,confidences_np,index)
    return result_img

# test
#img = io.imread('TEST/TEST.jpeg')
#results = yolo_predictions(img,net)
#io.imsave('./abc.jpeg', img)

# fig = px.imshow(img)
# fig.update_layout(width=700, height=400, margin=dict(l=10, r=10, b=10, t=10))
# fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
# fig.show()   