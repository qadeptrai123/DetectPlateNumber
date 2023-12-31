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
import preprocess

from glob import glob
from skimage import io
from shutil import copy
from paddleocr import PaddleOCR
ocr = PaddleOCR()
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.keras.applications import InceptionResNetV2
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array


#eo.download_and_install_model('craft', 'craft_mlt_25k.pth')
# settings
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# LOAD THE IMAGE
#img = io.imread('TEST/TEST.jpg')

#fig = px.imshow(img)
#fig.update_layout(width=700, height=400, margin=dict(l=10, r=10, b=10, t=10))
#fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
#fig.show()

# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('./Model3/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

pt.pytesseract.tesseract_cmd = r'C:\Users\beose\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
myconfig = r'--oem 3 --psm 11'
#myconfig += r' -l eng'
#myconfig += r' --c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'




# Chọn ngôn ngữ (ví dụ: tiếng Anh và tiếng Việt)
#languages = ['en', 'vi']

# Cấu hình tham số cho detector và recognizer
# detector_params = {'name': 'craft', 'weights': 'craft_mlt_25k.pth'}
# recognizer_params = {'name': 'latin'}

cropped_path = ""

# #Tạo đối tượng Reader
# reader = eo.Reader(
#     lang_list=['en', 'vi'],
#     #detector=detector_params,
#     #recognizer=recognizer_params
# )

# extrating text
def extract_text(image, bbox, ind):
    x,y,w,h = bbox
    roi = image[y-10:y+h+10, x-10:x+w+10]
    if not roi.size:
        #print(f"Empty ROI for index {ind}")
        return 0
    
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    height, width, _ = roi.shape
    if height < 88:
        c = 88/height
        roi = cv2.resize(roi, None, fx = c, fy = c)
    res = ocr.ocr(roi)

    if res is not None:
        txt = ''
        for line in res:
            if line is not None:
                for word_info in line:
                    txt += word_info[-1][0]
        if len(txt) < 3: 
            return 0
        #print(txt)
        
        io.imsave('./static/cropped/' + str(ind) + '.jpeg', roi)
        return 1
    else:
        return 0
    

def get_text(image, bbox):
    x,y,w,h = bbox
    roi = image[y-10:y+h+10, x-10:x+w+10]
    if not roi.size:
        #print(f"Empty ROI for index {ind}")
        return ''
   
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    height, width, _ = roi.shape
    if height < 88:
        c = 88/height
        roi = cv2.resize(roi, None, fx = c, fy = c)
    res = ocr.ocr(roi)
    
    #io.imsave('./static/cropped/' + str(ind) + '.jpeg', roi)
    if res is not None:
        txt = ''
        for line in res:
            if line is not None:
                for word_info in line:
                    txt += word_info[-1][0]
        if len(txt) < 3: 
            return ''
        return txt
    else:
        return ''

def get_detections(img, net):
    # 1.CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3), dtype=np.uint8)
    input_image[0:row, 0:col] = image[:, :, :3]

    # 2. GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH,INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    # print(preds[0])
    # print('\n')
    #print(detections)
    return input_image, detections

def non_maximum_supression(input_image,detections, type_input):

    # 3. FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE

    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        #print(row)
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



def drawings(image, boxes_np, confidences_np, index, type_input):
    # 5. Drawings
    cnt = 0
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        #conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        ok = extract_text(image, boxes_np[ind], cnt)
        #print("Plate is:" + license_text)
        
        cnt += 1
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),5)
        #cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,0),2)
        #cv2.rectangle(image,(x,y+h),(x+w,y+h+25),(255,0,0),2)

        #cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
        if type_input == 1:
            license_text = get_text(image, boxes_np[ind])
            cv2.putText(image, license_text, (x,y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    return image, cnt

# predictions flow with return result
def yolo_predictions(img, net, type_input):
    # step-1: detections
    input_image, detections = get_detections(img, net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections, type_input)
    # step-3: Drawings
    result_img, cnt = drawings(img,boxes_np,confidences_np,index, type_input)
    return result_img, cnt

# test
#img = io.imread('TEST/TEST.jpeg')
#results, cnt = yolo_predictions(img, net, 0)
#io.imsave('./RESULT/abc.jpeg', img)
#print(cnt)
# fig = px.imshow(img)
# fig.update_layout(width=700, height=400, margin=dict(l=10, r=10, b=10, t=10))
# fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
# fig.show()   

#video
def frame_to_jpeg(frame):
    # Chuyển đổi frame thành mảng bytes
    ret, jpeg = cv2.imencode('.jpg', frame)
    
    # Kiểm tra xem chuyển đổi có thành công không
    if not ret:
        return None
    
    # Chuyển đối tượng bytes thành mảng bytes thông thường
    return jpeg.tobytes()

def video_process(cap):
    #cap = cv2.VideoCapture('./TEST/test.mp4')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Có thể sử dụng 'mp4v' cho định dạng MP4
    out = cv2.VideoWriter('./static/output_video.mp4', fourcc, int(fps), (w, h))

    #read frames
    ret = True
    frame_nmr = -1
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            # Chuyển đổi frame thành ảnh JPEG
            jpeg_data = frame_to_jpeg(frame)
            
            # Lưu ảnh JPEG vào tệp tin
            with open('./TEST/output.jpeg', 'wb') as f:
                f.write(jpeg_data)
            img = io.imread('./TEST/output.jpeg')
            result, index = yolo_predictions(img, net, 1)
            #io.imsave('./RESULT/demo' + str(frame_nmr) + '.jpeg', img)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out.write(img)
    # cap.release()
    # out.release()