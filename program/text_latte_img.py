import cv2
import numpy as np


fonts = (cv2.FONT_HERSHEY_COMPLEX,
         cv2.FONT_HERSHEY_COMPLEX_SMALL,
         cv2.FONT_HERSHEY_DUPLEX,
         cv2.FONT_HERSHEY_PLAIN,
         cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
         cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
         cv2.FONT_HERSHEY_SIMPLEX,
         cv2.FONT_HERSHEY_TRIPLEX,
         cv2.FONT_ITALIC)


def create_circle_img(width, height):
    size = height, width, 3
    if height >= width:
        radius = int(width / 2)
    else:
        radius = int(height / 2)
    img_circle = np.zeros(size, dtype=np.uint8)
    cv2.circle(img_circle, (int(width/2), int(height/2)), radius, (128, 165, 220), -1)
    return img_circle


def test(img):
    cv2.imshow('test', img)
    cv2.waitKey(0)


def conver_latte(img, back_img):
    img_gauss = cv2.GaussianBlur(img, (5, 5), 0)
    img_blend = cv2.addWeighted(img_gauss, 0.5, back_img, 0.5, 0.0)
    return img_blend


def set_text_position_center(img_width, img_height, text, fontFace, fontScale, thickness):
    text_size, baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
    center_point = (int((img_width - text_size[0]) / 2), int((img_height + text_size[1]) / 2))
    return center_point


def create_text_latte_img(back_img_path, text):
    img_width = 380
    img_height = 375
    back_img = cv2.imread(back_img_path)
    back_img_resized = cv2.resize(back_img, (img_width, img_height))
    circle_img = create_circle_img(img_width, img_height)
    circle_img_thre = cv2.cvtColor(circle_img, cv2.COLOR_BGR2GRAY)
    back_img_resized = cv2.bitwise_and(back_img_resized, back_img_resized, mask = circle_img_thre)
    font = fonts[8]
    scale = 2
    thickness = 3
    text_position = set_text_position_center(img_width, img_height, text, font, scale, thickness)
    color = (200, 210, 240)
    cv2.putText(circle_img, text, text_position, font, scale, color, thickness, lineType=8, bottomLeftOrigin=False)
    latte_img = conver_latte(circle_img, back_img_resized)
    latte_img = cv2.GaussianBlur(latte_img, (5, 5), 0)
    test(latte_img)


if __name__ == '__main__':
    back_img_path = '../image/back_coffee1.jpg'
    create_text_latte_img(back_img_path, 'Kesuke')

