import cv2
import numpy as np
import time
import sys

fonts = (cv2.FONT_HERSHEY_COMPLEX,
         cv2.FONT_HERSHEY_COMPLEX_SMALL,
         cv2.FONT_HERSHEY_DUPLEX,
         cv2.FONT_HERSHEY_PLAIN,
         cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
         cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
         cv2.FONT_HERSHEY_SIMPLEX,
         cv2.FONT_HERSHEY_TRIPLEX,
         cv2.FONT_ITALIC)

def nothing(x):
    pass


def test(img):
    cv2.imshow('test', img)
    cv2.waitKey(0)


def process_back_image(img_src, back_img):
    height, width = img_src.shape[:2]
    back_img_resize = cv2.resize(back_img, (width, height))
    return back_img_resize


def create_monochromatic(r, g, b, width, height):
    size = height, width, 3
    contours = np.array([[0, 0], [0, height], [width, height], [width, 0]])
    color_img = np.zeros(size, dtype=np.uint8)
    cv2.fillPoly(color_img, pts=[contours], color=(b, g, r))
    return color_img


def create_circle_img(width, height):
    size = height, width, 1
    if height >= width:
        radius = int(width / 2)
    else:
        radius = int(height / 2)
    img_circle = np.zeros(size, dtype=np.uint8)
    cv2.circle(img_circle, (int(width/2), int(height/2)), radius, (255, 255, 255), -1)
    return img_circle


def convert_latte(img_thresh, back_img, brown_img, white_img, circle_img):
    img_thresh_inv = cv2.bitwise_not(img_thresh)
    img_brown_masked = cv2.bitwise_and(brown_img, brown_img, mask=img_thresh)
    img_white_masked = cv2.bitwise_and(white_img, white_img, mask=img_thresh_inv)
    img_color = cv2.bitwise_or(img_brown_masked, img_white_masked)
    img_color_gauss = cv2.GaussianBlur(img_color,(5,5),0)
    img_blend = cv2.addWeighted(img_color_gauss, 0.5, back_img, 0.5, 0.0)
    img_latte = cv2.bitwise_and(img_blend, img_blend, mask=circle_img)
    return img_latte


def get_latte_img_from_path(img_path, back_img_path):
    cv2.namedWindow('original', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('adapt_mean_window', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('adapt_gaussian_window', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('adapt_mean_rough_window', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('adapt_gaussian_rough_window', cv2.WINDOW_AUTOSIZE)
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


def get_latte_img_from_cam(img, back_img, width, height, brown_img, white_img, circle_img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_mean_thre = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 2)
    img_gaussian_thre = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2)
    img_mean_rough_thre = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 2)
    img_gaussian_rough_thre = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                                    21, 2)
    img_mean_latte = convert_latte(img_mean_thre, back_img, brown_img, white_img, circle_img)
    img_gaussian_latte = convert_latte(img_gaussian_thre, back_img, brown_img, white_img, circle_img)
    img_mean_rough_latte = convert_latte(img_mean_rough_thre, back_img, brown_img, white_img, circle_img)
    img_gaussian_rough_latte = convert_latte(img_gaussian_rough_thre, back_img, brown_img, white_img, circle_img)
    cv2.imshow('adapt_mean_window', img_mean_latte)
    cv2.imshow('adapt_gaussian_window', img_gaussian_latte)
    cv2.imshow('adapt_mean_rough_window', img_mean_rough_latte)
    cv2.imshow('adapt_gaussian_rough_window', img_gaussian_rough_latte)


def get_latte_img_from_cam_thresh(img, back_img, width, height):
    thresh = 100
    cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
    back_img_processed = cv2.resize(back_img, (width, height))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img_thresh = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    img_thresh_latte = convert_latte(img_thresh, back_img_processed)
    cv2.imshow('thresh', img_thresh_latte)


def camera_img_processing(back_img_path):
    cv2.namedWindow('original', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('adapt_mean_window', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('adapt_gaussian_window', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('adapt_mean_rough_window', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('adapt_gaussian_rough_window', cv2.WINDOW_AUTOSIZE)
    counter = 0
    camera_width = 380
    camera_height = 375
    back_img = cv2.imread(back_img_path)
    back_img_processed = cv2.resize(back_img, (camera_width, camera_height))
    brown_img = create_monochromatic(128, 65, 50, camera_width, camera_height)
    white_img = create_monochromatic(245, 245, 255, camera_width, camera_height)
    circle_img = create_circle_img(camera_width, camera_height)
    start_time = time.time()
    cap = cv2.VideoCapture(0)
    if cap.isOpened() is False:
        print("error to connect camera")
        sys.exit()
    cap.set(3, camera_width)
    cap.set(4, camera_height)
    while True:
        ret, img = cap.read()
        if ret == False:
            continue
        cv2.imshow('original', img)
        counter += 1
        counter %= 60
        get_latte_img_from_cam(img, back_img_processed, camera_width, camera_height, brown_img, white_img, circle_img)
        # get_latte_img_from_cam_thresh(img, back_img, camera_width, camera_height)
        now_time = time.time()
        fps = round(1 / (now_time - start_time), 1)
        fps_position = (0, 30)
        font = fonts[8]
        size = 1
        color = (255, 255, 255)
        start_time = time.time()
        face_detection(img)
        cv2.putText(img, str(fps) + '[fps]', fps_position, font, size, color)
        cv2.imshow('original', img)
        key = cv2.waitKey(1)
        if key == 0x1b:
            break
    cap.release()
    cv2.destroyAllWindows()


def face_detection(img):
    color = (255, 255, 255)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    facerect = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    if len(facerect) > 0:
        for rect in facerect:
            cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color, thickness=2)

if __name__ == '__main__':
    back_img_path = '../image/back_coffee1.jpg'
    camera_img_processing(back_img_path)