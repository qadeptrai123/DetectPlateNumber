from flask import Flask, render_template, request
import os
import main
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
        results = main.yolo_predictions(img, main.net)
        main.io.imsave('./static/done/'+filename, img)
        #trích xuất chữ
        roi = main.io.imread(main.cropped_path)
        grayimage = main.cv2.cvtColor(roi, main.cv2.COLOR_BGR2GRAY)
        #grayimage = main.cv2.convertScaleAbs(grayimage, alpha=0.5, beta=0)
        main.io.imsave('./ttt.jpeg', grayimage)
        text = main.pt.image_to_string(grayimage, config=main.myconfig)
        text = text.strip()
    
        return render_template('index.html', upload=True, upload_image=filename, cropped_image=filename, text=text)

    return render_template('index.html')

if __name__ =="__main__":
    app.run(debug=True)