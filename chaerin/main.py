import cv2
import dlib
import sys
import numpy as np
from tkinter import *
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
# from PyQt5.QtGui import QIcon # 위젯창 아이콘
scaler = 1

overlay = cv2.imread('beauty_samples/ryan_transparent.png',
                     cv2.IMREAD_UNCHANGED)


def button1_clicked():
    global overlay
    overlay = cv2.imread('beauty_samples/ryan_transparent.png',
                         cv2.IMREAD_UNCHANGED)
    snow_camera()


def button2_clicked():
    global overlay
    overlay = cv2.imread('beauty_samples/cute_mask.png',
                         cv2.IMREAD_UNCHANGED)
    snow_camera()


def button3_clicked():
    global overlay
    overlay = cv2.imread('beauty_samples/heavy_mask.png',
                         cv2.IMREAD_UNCHANGED)
    snow_camera()


app = QApplication(sys.argv)  # application 객체 생성

w = QWidget()  # default Interface 선언
w.setWindowTitle('Button')  # 위젯 제목 설정
w.setGeometry(300, 300, 300, 300)  # 위젯 크기

button1 = QPushButton('ryan', w)  # 버튼 추가
button1.resize(100, 100)  # 버튼 크기
button1.move(0, 100)  # 버튼 위치
button1.clicked.connect(button1_clicked)

button2 = QPushButton('cute', w)  # 버튼 추가
button2.resize(100, 100)  # 버튼 크기
button2.move(100, 100)  # 버튼 위치
button2.clicked.connect(button2_clicked)

button3 = QPushButton('gas_mask', w)  # 버튼 추가
button3.resize(100, 100)  # 버튼 크기
button3.move(200, 100)  # 버튼 위치
button3.clicked.connect(button3_clicked)

w.show()  # 스크린에 띄우기


face_cascade = cv2.CascadeClassifier(
    'FACE_DETECTOR_MASTER/haarcascade_frontalface_default.xml')
# Read the input image
#img = cv2.imread('test.png')


def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(),
                              mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img


face_roi = []
face_sizes = []


def snow_camera():
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('samples/girl.mp4')
    while cap.isOpened():
        _, img = cap.read()
        img = cv2.resize(
            img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:

            result = overlay_transparent(
                img, overlay, (2*x+w)/2, (2*y+h)/2, overlay_size=(w, h))
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        # Display the output
        cv2.imshow('img', img)
        cv2.imshow('result', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


sys.exit(app.exec_())
