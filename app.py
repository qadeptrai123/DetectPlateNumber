from flask import Flask, render_template, request
import os
import main
import preprocess
from paddleocr import PaddleOCR

ocr = PaddleOCR()

# Webserver gateway interface
app = Flask (__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        main.cropped_path = "./static/cropped/" + filename
        img = main.io.imread('./static/upload/'+filename)
        results, cnt = main.yolo_predictions(img, main.net)
        main.io.imsave('./static/done/'+filename, img)
        #trích xuất chữ
        #text2 = preprocess.preprocess(roi)
        #grayimage = main.cv2.cvtColor(roi, main.cv2.COLOR_BGR2GRAY)
        
        #grayimage = main.cv2.convertScaleAbs(grayimage, alpha=0.5, beta=0)
        #grayimage = main.cv2.GaussianBlur(grayimage, (5, 5), 0)
       # _, grayimage = main.cv2.threshold(grayimage, 128, 255, main.cv2.THRESH_BINARY)
        # clahe = main.cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # grayimage = clahe.apply(grayimage)
        #grayimage = main.cv2.convertScaleAbs(grayimage, alpha=1.5, beta=3)
        #grayimage = main.cv2.equalizeHist(grayimage)
        # h, w, _ = img.shape
        # new_h = int(600 * (600 / w))
        # new_w = int(600)
        # grayimage = main.cv2.resize(grayimage, (new_w, new_h))
        # main.io.imsave('./ttt.jpeg', grayimage)
        # text = main.pt.image_to_string(grayimage, config=main.myconfig)
        #text2.strip()
        # lines = text.splitlines()
        # res = ""
        # for line in lines:
        #     res = res + line
        # res = res.strip()

        #process each cropped image
        plate_list = []
        for index in range(cnt):
            #lấy ảnh cropped
            roi = main.io.imread('./static/cropped/' + str(index+1) + '.jpeg')
            #paddle OCR
            height, width, _ = roi.shape
            if height < 88:
                c = 88/height
                roi = main.cv2.resize(roi, None, fx = c, fy = c)
            #print(roi.shape())
            text = ''
            res = ocr.ocr(roi)
            if res is not None:
                for line in res:
                    if line is not None:
                        for word_info in line:
                            if len(word_info[-1][0]) >= 3:
                                text += word_info[-1][0]
            #print(text + '\n')
            plate_list.append(text)
            

        return render_template('index.html', upload=True, upload_image=filename, plate_list=plate_list)

    return render_template('index.html')

if __name__ =="__main__":
    app.run(debug=True)