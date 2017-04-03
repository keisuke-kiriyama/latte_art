import cv2
import time

img_path = '../image/fukakyon_1.jpg'
cv2.namedWindow('threshold', cv2.WINDOW_NORMAL)
cv2.namedWindow('mean_threshold', cv2.WINDOW_NORMAL)
cv2.namedWindow('gaussian_threshold', cv2.WINDOW_NORMAL)
img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
threshold_start = time.time()
thresh_img = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
threshold_end = time.time()
mean_start = time.time()
adapt_thresh_img = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
mean_end = time.time()
gauss_start = time.time()
cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
gauss_end = time.time()

print('threshold',(threshold_end - threshold_start))
print('mean_threshold', (mean_end - mean_start))
print('gaussian_threshold', (gauss_end - gauss_start))
