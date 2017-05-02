import cv2
import numpy as np
from flask import Flask, request, jsonify, helpers
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

app = Flask(__name__)
app.config.from_object(__name__)


def draw_text_at_center(text, img_width, img_height):
    img = Image.new("RGB", (img_width, img_height), (191, 236, 255))
    draw = ImageDraw.Draw(img)
    draw.font = ImageFont.truetype("JKG-L_3.ttf", 20)
    img_size = np.array(img.size)
    txt_size = np.array(draw.font.getsize(text))
    pos = (img_size - txt_size) / 2
    draw.text(pos, text, (0, 56, 135))
    return img

def convert_latte(text_mat, back_mat):
    img_gauss = cv2.GaussianBlur(text_mat, (5, 5), 0)
    img_blend = cv2.addWeighted(img_gauss, 0.5, back_mat, 0.5, 0.0)
    return img_blend


def create_text_latte_img(back_img_path, text):
    img_width = 100
    img_height = 100
    back_mat = cv2.imread(back_img_path)
    back_mat = cv2.resize(back_mat, (img_width, img_height))
    text_img = draw_text_at_center(text, img_width, img_height)
    text_mat = np.asarray(text_img)
    latte_mat = convert_latte(text_mat, back_mat)
    latte_mat = cv2.cvtColor(latte_mat, cv2.COLOR_RGB2BGR)
    latte_img = Image.fromarray(np.uint8(latte_mat))
    return latte_img


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "ok"})


@app.route('/text', methods=['POST'])
def create_text_latte():
    text = request.form['text']
    latte_img = create_text_latte_img('../image/back_coffee1.jpg', text)
    buf = BytesIO()
    latte_img.save(buf, 'png')
    response = helpers.make_response(buf.getvalue())
    response.headers["Content-Type"] = "Image/png"
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0')
