import cv2

cap = cv2.VideoCapture('movie.mp4')
while True:
    ret, img = cap.read()
    cv2.imshow('test', img)
    cv2.waitKey(1)

