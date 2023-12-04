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
        a, b = filename.split('.')
        if b == 'mp4' or b == 'avi':
            cap = main.cv2.VideoCapture('./static/upload/' + filename)
            #main.video_process(cap)
            return render_template('index.html', upload=2)
        
        main.cropped_path = "./static/cropped/" + filename
        img = main.io.imread('./static/upload/'+filename)
        results, cnt = main.yolo_predictions(img, main.net, 0)
        main.io.imsave('./static/done/'+filename, img)

        #process each cropped image
        plate_list = []
        for index in range(cnt):
            #lấy ảnh cropped
            roi = main.io.imread('./static/cropped/' + str(index) + '.jpeg')
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
                            if len(word_info[-1][0]) > 0:
                                text += word_info[-1][0]
                            #print(word_info[-1][0])
            #print(text + '\n')
            
            roi = main.cv2.cvtColor(roi, main.cv2.COLOR_BGR2RGB)
            main.io.imsave('./static/cropped/' + str(index) + '.jpeg', roi)
            plate_list.append(text)
            

        return render_template('index.html', upload=1, upload_image=filename, plate_list=plate_list)

    return render_template('index.html')

if __name__ =="__main__":
    app.run(debug=True)