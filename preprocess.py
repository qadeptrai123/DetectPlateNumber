import pytesseract
import cv2
import os
from skimage import io
import numpy as np
import easyocr as eo

detector_params = {'name': 'craft', 'weights': 'craft_mlt_25k.pth'}
recognizer_params = {'name': 'latin'}

reader = eo.Reader(
    lang_list=['en'],
    detector=detector_params,
    recognizer=recognizer_params
)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\beose\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
# point to license plate image (works well with custom crop function)
def preprocess(gray):
    #gray = cv2.imread("./detections/crop/car3/license_plate_.png", 0)
    gray = cv2.resize( gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    # blur = cv2.GaussianBlur(gray, (5,5), 0)
    # gray = cv2.medianBlur(gray, 3)
    io.imsave("./TEST/gray.jpeg", gray)
    blur = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    io.imsave("./TEST/blur.jpeg", blur)
    # perform otsu thresh (using binary inverse since opencv contours work better with white text)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    #cv2.imshow("Otsu", thresh)
    #cv2.waitKey(0)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    io.imsave("./TEST/thresh.jpeg", thresh)
    # apply dilation 
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    #cv2.imshow("dilation", dilation)
    #cv2.waitKey(0)
    dilation = cv2.bitwise_not(dilation)
    # find contours
    
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
   # io.imsave("./TEST/contours.jpeg", hierarchy);
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    io.imsave("./TEST/dilation.jpeg", dilation)
    # create copy of image
    im2 = gray.copy()

    plate_num = ""
    cnt1 = 0
    #loop through contours and find letters in license plate
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width, _ = im2.shape
        cnt1 += 1
        # if height of box is not a quarter of total height then skip
        if height / float(h) > 6: continue
        ratio = h / float(w)
        # if height to width ratio is less than 1.5 skip
        if ratio < 1.5: continue
        area = h * w
        # if width is not more than 25 pixels skip
        #if width / float(w) > 15: continue
        # if area is less than 100 pixels skip
        if area < 100: continue
        # draw the rectangle
        rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        roi = cv2.bitwise_not(roi)
        roi = cv2.medianBlur(roi, 5)
        #cv2.imshow("ROI", roi)
        #cv2.waitKey(0)
        p = 'roi' + str(cnt1) 
        io.imsave('./TEST/' + p + '.jpeg', roi)
        #text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10 --oem 3')
        #text = pytesseract.image_to_string(roi)
        #print(cnt1)
        #print(' ')
        #print(text) 
        text = reader.readtext(roi)
        res = ""
        for dect in text:
            res = res + dect[1]
        plate_num += res
    #text = pytesseract.image_to_string(dilation, config=r'--psm 11 --oem 3')
    # text = reader.readtext(dilation)
    # res = ""
    # for dect in text:
    #     res = res + dect[1]
    io.imsave("./TEST/im2.jpeg", im2)
    return plate_num