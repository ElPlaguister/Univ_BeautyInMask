# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import sys
from tkinter import*
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon  # 위젯창 아이콘

scaler = 1
result = 0
overlay = cv2.imread('samples/1.png',
                     cv2.IMREAD_UNCHANGED)
previous_mask_status = True


def button0_clicked():  # non-filter
    global overlay

    overlay = cv2.imread('samples/0.png', cv2.IMREAD_UNCHANGED)
    filter_camera()


def button1_clicked():
    global overlay

    overlay = cv2.imread('samples/1.png', cv2.IMREAD_UNCHANGED)
    filter_camera()


def button2_clicked():
    global overlay

    overlay = cv2.imread('samples/2.png', cv2.IMREAD_UNCHANGED)
    filter_camera()


def button3_clicked():
    global overlay

    overlay = cv2.imread('samples/3.png', cv2.IMREAD_UNCHANGED)
    filter_camera()


def button4_clicked():
    global overlay

    overlay = cv2.imread('samples/4.png', cv2.IMREAD_UNCHANGED)
    filter_camera()


def buttonQuit_clicked():  # 종료버튼
    sys.exit()
    # buttonQuit.clicked.connect(QCoreApplication.instance().quit)


app = QApplication(sys.argv)  # application 객체 생성
app.setWindowIcon(QIcon('samples/icon.png'))  # 위젯창 아이콘 설정

w = QWidget()  # default Interface 선언
w.setWindowTitle('Filter Camera')  # 위젯 제목 설정
w.setGeometry(100, 100, 500, 150)  # 위젯 크기

button0 = QPushButton('X', w)  # 'X' 가 쓰여진 버튼 추가
button0.resize(100, 100)  # 버튼 크기
button0.move(0, 0)  # 버튼 위치
button0.setStyleSheet("font-size:80px;")  # 폰트
button0.setToolTip('non-filter')  # 버튼 툴 팁
button0.clicked.connect(button0_clicked)

button1 = QPushButton('', w)
button1.resize(100, 100)
button1.move(100, 0)
button1.setIcon(QIcon('BUTTON/samples/1.png'))  # 버튼 이미지
button1.setIconSize(QSize(90, 90))  # 버튼 이미지 크기
button1.clicked.connect(button1_clicked)

button2 = QPushButton('', w)
button2.setIcon(QIcon('BUTTON/samples/2.png'))
button2.setIconSize(QSize(90, 90))
button2.resize(100, 100)
button2.move(200, 0)
button2.clicked.connect(button2_clicked)

button3 = QPushButton('', w)
button3.resize(100, 100)
button3.move(300, 0)
button3.setIcon(QIcon('BUTTON/samples/3.png'))
button3.setIconSize(QSize(90, 90))
button3.clicked.connect(button3_clicked)

button4 = QPushButton('', w)
button4.resize(100, 100)
button4.move(400, 0)
button4.setIcon(QIcon('BUTTON/samples/4.png'))
button4.setIconSize(QSize(90, 90))
button4.clicked.connect(button4_clicked)

buttonQuit = QPushButton('Quit', w)
buttonQuit.resize(100, 30)
buttonQuit.move(200, 110)
buttonQuit.setStyleSheet(
    "background-color: red;font-size:20px;font-family:Times New Roman;")
buttonQuit.clicked.connect(buttonQuit_clicked)

w.show()  # 스크린에 띄우기

face_cascade = cv2.CascadeClassifier(
    'samples/haarcascade_frontalface_default.xml')
# Read the input image
# img = cv2.imread('test.png')


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

    bg_img[int(y-h/2):int(y+h/2), int(x-w/2)           :int(x+w/2)] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img


face_roi = []
face_sizes = []


def filter_camera():
    global result
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('samples/girl.mp4')
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
        # cv2.imshow('img', img)
        cv2.imshow('result', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        if label == "Mask" and previous_mask_status == False:
            button3_clicked()
        elif label == "No Mask" and previous_mask_status == True:
            button4_clicked()

    # 전 프레임의 마스크 상황
        if label == "Mask":
            previous_mask_status = True
        else:
            previous_mask_status = False

       # display the label and bounding box rectangle on the output
       # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
sys.exit(app.exec_())
