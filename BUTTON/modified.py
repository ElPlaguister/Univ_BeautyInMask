import cv2
from tkinter import*
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon #위젯창 아이콘

scaler = 1
result = 0
overlay = cv2.imread('beauty_samples/ryan_transparent.png',cv2.IMREAD_UNCHANGED)

def button0_clicked(): #non-filter
    global overlay

    overlay = cv2.imread('samples/0.png', cv2.IMREAD_UNCHANGED)
    filter_camera()

def button1_clicked():
    global overlay

    overlay = cv2.imread('samples/1.png',cv2.IMREAD_UNCHANGED)
    filter_camera()

def button2_clicked():
    global overlay

    overlay = cv2.imread('samples/2.png',cv2.IMREAD_UNCHANGED)
    filter_camera()

def button3_clicked():
    global overlay

    overlay = cv2.imread('samples/3.png',cv2.IMREAD_UNCHANGED)
    filter_camera()

def button4_clicked():
    global overlay

    overlay = cv2.imread('samples/4.png',cv2.IMREAD_UNCHANGED)
    filter_camera()

def buttonQuit_clicked(): #종료버튼
    sys.exit()
    #buttonQuit.clicked.connect(QCoreApplication.instance().quit)

app = QApplication(sys.argv)  # application 객체 생성
app.setWindowIcon(QIcon('samples/icon.png')) #위젯창 아이콘 설정

w = QWidget()  # default Interface 선언
w.setWindowTitle('Filter Camera')  # 위젯 제목 설정
w.setGeometry(100, 100, 500, 150)  # 위젯 크기

button0 = QPushButton('X', w)  # 'X' 가 쓰여진 버튼 추가
button0.resize(100, 100)  # 버튼 크기
button0.move(0, 0)  # 버튼 위치
button0.setStyleSheet("font-size:80px;") # 폰트
button0.setToolTip('non-filter') # 버튼 툴 팁
button0.clicked.connect(button0_clicked)

button1 = QPushButton('', w)
button1.resize(100, 100)
button1.move(100, 0)
button1.setIcon(QIcon('samples/1.png')) # 버튼 이미지
button1.setIconSize(QSize(90, 90)) # 버튼 이미지 크기
button1.clicked.connect(button1_clicked)

button2 = QPushButton('', w)
button2.setIcon(QIcon('samples/2.png'))
button2.setIconSize(QSize(90, 90))
button2.resize(100, 100)
button2.move(200, 0)
button2.clicked.connect(button2_clicked)

button3 = QPushButton('', w)
button3.resize(100, 100)
button3.move(300, 0)
button3.setIcon(QIcon('samples/3.png'))
button3.setIconSize(QSize(90, 90))
button3.clicked.connect(button3_clicked)

button4 = QPushButton('', w)
button4.resize(100, 100)
button4.move(400, 0)
button4.setIcon(QIcon('samples/4.png'))
button4.setIconSize(QSize(90, 90))
button4.clicked.connect(button4_clicked)

buttonQuit=QPushButton('Quit', w)
buttonQuit.resize(100, 30)
buttonQuit.move(200, 110)
buttonQuit.setStyleSheet("background-color: red;font-size:20px;font-family:Times New Roman;")
buttonQuit.clicked.connect(buttonQuit_clicked)

w.show() #스크린에 띄우기

face_cascade = cv2.CascadeClassifier('samples/haarcascade_frontalface_default.xml')
#Read the input image
#img = cv2.imread('test.png')

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    #convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

    #convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img

face_roi = []
face_sizes = []

def filter_camera():
    global result
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('samples/girl.mp4')
    while cap.isOpened():
        _, img = cap.read()
        img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            result = overlay_transparent(img, overlay, (2*x+w)/2, (2*y+h)/2, overlay_size=(w, h))
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

        #Display the output
        #cv2.imshow('img', img)
        cv2.imshow('result', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

sys.exit(app.exec_())