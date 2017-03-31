import cv2
import numpy as np


def nothing(x):
    pass


def process_back_image(img_src, back_img):
    height, width = img_src.shape[:2]
    back_img_resize = cv2.resize(back_img, (width, height))
    return back_img_resize


def convert_latte(img_thre, back_img):
    height, width = img_thre.shape[:2]
    size = height, width, 3
    if height >= width:
        radius = int(width / 2)
    else:
        radius = int(height / 2)
    img_color = np.zeros(size, dtype=np.uint8)
    cv2.circle(img_color, (int(width/2), int(height/2)), radius, (255, 255, 255), -1)
    for y in range(0, height):
        for x in range(0, width):
            if((img_color[y, x] == [255, 255, 255]).all()):
                if (img_thre[y, x] == 0):
                    img_color[y, x] = [50, 65, 128]
                else:
                    img_color[y, x] = [245, 245, 255]
            else:
                back_img[y, x] = [0, 0, 0]
    img_color_gauss = cv2.GaussianBlur(img_color,(5,5),0)
    img_blend = cv2.addWeighted(img_color_gauss, 0.5, back_img, 0.5, 0.0)
    return img_blend


def get_latte_img(img_path, back_img_path):
    cv2.namedWindow('original', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('adapt_mean_window', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('adapt_gaussian_window', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('adapt_mean_rough_window', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('adapt_gaussian_rough_window', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('thresh', 'thresh_window', 0, 255, nothing)
    cv2.setTrackbarPos('thresh', 'thresh_window', 100)
    img_src = cv2.imread(img_path)
    back_img = cv2.imread(back_img_path)
    back_img_processed = process_back_image(img_src, back_img)
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
    img_mean_thre = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
    img_gaussian_thre = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    img_mean_rough_thre = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)
    img_gaussian_rough_thre = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    img_mean_latte = convert_latte(img_mean_thre, back_img_processed)
    img_gaussian_latte = convert_latte(img_gaussian_thre, back_img_processed)
    img_mean_rough_latte = convert_latte(img_mean_rough_thre, back_img_processed)
    img_gaussian_rough_latte = convert_latte(img_gaussian_rough_thre, back_img_processed)
    cv2.imshow('original', img_src)
    cv2.imshow('adapt_mean_window', img_mean_latte)
    cv2.imshow('adapt_gaussian_window', img_gaussian_latte)
    cv2.imshow('adapt_mean_rough_window', img_mean_rough_latte)
    cv2.imshow('adapt_gaussian_rough_window', img_gaussian_rough_latte)
    cv2.waitKey()

if __name__ == '__main__':
    img_path = '../image/fukakyon_1.jpg'
    back_img_path = '../image/back_coffee1.jpg'
    get_latte_img(img_path, back_img_path)