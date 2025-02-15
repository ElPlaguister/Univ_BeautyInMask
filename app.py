from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import os.path
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon

overlay_mask = cv2.imread("set_up/filter/default.png", cv2.IMREAD_UNCHANGED)
overlay_nonMask = cv2.imread("set_up/filter/default.png", cv2.IMREAD_UNCHANGED)

test_mode = True
filter_scale = 2

def get_img_path(index = 0, maskMode = False, imtype = 'png'):
    mask_keyword = "mask" if maskMode else "non_mask"
    path = os.path.sep.join(["set_up", "filter", str(index), mask_keyword + "." + imtype])
    return path

def get_img(index = 0, maskMode = False, imtype = 'png'):
    img = cv2.imread(get_img_path(index = index, maskMode = maskMode, imtype = imtype), cv2.IMREAD_UNCHANGED)
    return img
    
def exit():
    print("끝")
    cv2.destroyAllWindows()
    vs.stop()
    sys.exit()

class BeautyButton():
    myWidget = None
    buttonSize = 0
    def __init__(self, index, character = ''):
        # 창을 초기화하지 않았다면 새로 만들어줍니다.
        if BeautyButton.myWidget == None:
            BeautyButton.init_widget()
        # 버튼 번호에 맞는 이미지들을 등록해줍니다.
        self.maskImage = get_img(index, True)
        self.nonMaskImage = get_img(index, False)
        self.button = QPushButton(character, BeautyButton.myWidget)
        self.button.resize(100, 100)
        self.button.move(BeautyButton.buttonSize * 100, 0)
        self.index = BeautyButton.buttonSize
        self.button.clicked.connect(self.myClick)
        BeautyButton.buttonSize += 1
        if character == '':
            self.button.setIcon(QIcon(get_img_path(index, True)))
            self.button.setIconSize(QSize(90, 90))
        else:
            self.button.setStyleSheet("font-size:80px;")
            self.button.setToolTip('non-filter')
    
    def myClick(self):
        global overlay_mask, overlay_nonMask
        print("버튼 클릭: {}".format(self.index))
        overlay_mask = self.maskImage
        overlay_nonMask = self.nonMaskImage
        
    @staticmethod
    def add_quit_button():
        quit_button = QPushButton('Quit', BeautyButton.myWidget)
        quit_button.resize(100, 30)
        quit_button.move(200, 110)
        quit_button.setStyleSheet("background-color: red;font-size:20px;font-family:Times New Roman;")
        quit_button.clicked.connect(exit)
    
    @staticmethod
    def init_widget(filter_size = 5):
        BeautyButton.myWidget = QWidget()
        BeautyButton.myWidget.setWindowTitle('Filter Camera')
        BeautyButton.myWidget.setGeometry(100, 100, filter_size * 100, 150)
    
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

    return (locs, preds)

if __name__ == "__main__":
    app = QApplication(sys.argv)  # application 객체 생성
    app.setWindowIcon(QIcon('samples/icon.png'))  # 위젯창 아이콘 설정
    
    buttons = []
    for i in range(0, 100, 1):
        path = os.path.sep.join(["set_up", "filter", str(i)])
        if os.path.isdir(path):
            buttons.append(BeautyButton(i, ('X' if i == 0 else '')))
        else:
            break
    
    BeautyButton.start_widget()
    
    face_cascade = cv2.CascadeClassifier('set_up/haarcascade_frontalface_default.xml')
    
    face_roi = []
    face_sizes = []
    
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join(["set_up", "face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["set_up", "face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(os.path.sep.join(["set_up", "mask_detector.model"]))

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    buttons[-1].myClick()
    
    # 실시간 비디오의 각 프레임마다 처리하는 부분입니다.
    while True:
        # 프레임을 받아오고 프레임 별 얼굴을 인식해 faces리스트에 담는 부분입니다.
        frame = vs.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        frame = imutils.resize(frame, width=400)
        
        # 얼굴들이 마스크를 쓴 상태인지 확인하는 함수를 호출합니다.
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        
        # 각 얼굴 별로 마스크 확률 및 위치를 이용해 적절한 이미지를 붙이는 부분입니다.
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            
            isMask = mask > withoutMask
            
            
            # 얼굴에 넣을 필터의 생김새와 크기를 설정하는 부분입니다.
            overlay = overlay_mask if isMask else overlay_nonMask      
            
            filter_size = int((endX-startX)* filter_scale)
            frame = overlay_transparent(frame, overlay, (startX+endX)/2, (startY+endY)/2, overlay_size=(filter_size, filter_size))
            
            # 테스트모드인 경우 마스크 착용 확률을 포함하는 테스트용 레이아웃을 추가로 출력합니다.
            if test_mode:
                prob = max(mask, withoutMask)
                label = "Mask" if isMask else "No Mask"
                label = "{}: {:.2f}%".format(label, prob * 100)
                color = (0, 255, 0) if isMask else (0, 0, 255)
                
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # 결과 화면을 창에 띄웁니다.
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            exit()