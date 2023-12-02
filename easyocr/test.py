#pip install easyocr

import easyocr

reader = easyocr.Reader(['en'])

for img in ...:
    result = a[index]
    result_check = reader.readtext('./easyocr/image/im2.jpeg')
    text = ''
    for detection in result:
        text += detection[1]
    if text == result: cnt += 1

