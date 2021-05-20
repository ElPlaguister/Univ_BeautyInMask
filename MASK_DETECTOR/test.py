from six import int2byte
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import sys
from tkinter import*
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from tensorflow.python.saved_model.nested_structure_coder import _Int64Codec  # 위젯창 아이콘

scaler = 1
result = 0
overlay_mask = cv2.imread('samples/1.png', cv2.IMREAD_UNCHANGED)
overlay_nonMask = cv2.imread('samples/1.png', cv2.IMREAD_UNCHANGED)
overlay = cv2.imread('samples/1.png',
                     cv2.IMREAD_UNCHANGED)
previous_mask_status = True  # 마스크를 썼는지 아닌지
filter_num = 0
    
def exit():
    print("끝")
    sys.exit()

class BeautyButton():
    myWidget = None
    buttonSize = 0
    def __init__(self, maskImage, nonMaskImage, character = ''):
        if BeautyButton.myWidget == None:
            BeautyButton.init_widget()
        self.maskImage = cv2.imread(maskImage, cv2.IMREAD_UNCHANGED)
        self.nonMaskImage = cv2.imread(nonMaskImage, cv2.IMREAD_UNCHANGED)
        self.button = QPushButton(character, BeautyButton.myWidget)
        self.button.resize(100, 100)
        self.button.move(BeautyButton.buttonSize * 100, 0)
        self.index = BeautyButton.buttonSize
        self.button.clicked.connect(self.myClick)
        BeautyButton.buttonSize += 1
        if character == '':
            self.button.setIcon(QIcon(maskImage))
            self.button.setIconSize(QSize(90, 90))
        else:
            self.button.setStyleSheet("font-size:80px;")
            self.button.setToolTip('non-filter')
    
    def myClick(self):
        global overlay, overlay_mask, overlay_nonMask, filter_num
        print("버튼 클릭: {}".format(self.index))
        filter_num = self.index
        overlay_mask = self.maskImage
        overlay_nonMask = self.nonMaskImage
        overlay = self.maskImage
        
    @staticmethod
    def add_quit_button():
        quit_button = QPushButton('Quit', BeautyButton.myWidget)
        quit_button.resize(100, 30)
        quit_button.move(200, 110)
        quit_button.setStyleSheet("background-color: red;font-size:20px;font-family:Times New Roman;")
        quit_button.clicked.connect(exit)
    
    @staticmethod
    def init_widget():
        BeautyButton.myWidget = QWidget()
        BeautyButton.myWidget.setWindowTitle('Filter Camera')
        BeautyButton.myWidget.setGeometry(100, 100, 500, 150)
    
    @staticmethod
    def start_widget():
        BeautyButton.add_quit_button()
        BeautyButton.myWidget.show()
    
    

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
        min_confidence = 0.5

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > min_confidence:
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

if __name__ == "__main__":
    app = QApplication(sys.argv)  # application 객체 생성
    app.setWindowIcon(QIcon('samples/icon.png'))  # 위젯창 아이콘 설정
    
    buttons = [BeautyButton('samples/0.png', 'samples/0.png', 'X'),
    BeautyButton('samples/1.png', 'samples/1.png'),
    BeautyButton('samples/2.png', 'samples/2.png'),
    BeautyButton('samples/3.png', 'samples/4.png'),
    BeautyButton('samples/4.png', 'samples/3.png')]
    
    BeautyButton.start_widget()
    
    face_cascade = cv2.CascadeClassifier('samples/haarcascade_frontalface_default.xml')
    
    face_roi = []
    face_sizes = []
    
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model("mask_detector.model")

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
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

            if label == "Mask" and previous_mask_status == False:
                overlay = overlay_mask
            elif label == "No Mask" and previous_mask_status == True:
                overlay = overlay_nonMask

            # 전 프레임의 마스크 상황
            previous_mask_status = label == "Mask"

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            filter_scale = 1
            filter_size = int((endX-startX)* filter_scale)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            frame = overlay_transparent(
                frame, overlay, (startX+endX)/2, (startY+endY)/2, overlay_size=(filter_size, filter_size))
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



