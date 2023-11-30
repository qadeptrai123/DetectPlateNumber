import pytesseract as pt
from skimage import io

img = io.imread('./static/cropped/bien-so-vip-35a-333-33-o-ninh-binh-la-gia-hinh-anh023125954320230214153424-16765230296762047346674.jpeg')

text = pt.image_to_string(img, config='--oem 3')

print(text)