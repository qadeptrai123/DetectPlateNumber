from ultralytics import YOLO
import cv2

#load model yolov8 để bắt biển số trong video
coco_model = YOLO('yolov5n.pt')
license_plate_detector = YOLO('./Model3/weights/best.pt')

#load video
cap = cv2.VideoCapture('./TEST/TEST.mp4')

#read frames
ret = True
frame_nmr = -1
while ret and frame_nmr < 10:
  frame_nmr += 1
  ret, frame = cap.read()
  if ret:
    detections = coco_model(frame)
    print(detections)