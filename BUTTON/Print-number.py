import cv2, dlib, sys
from tkinter import *
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
#from PyQt5.QtGui import QIcon # 위젯창 아이콘

def button1_clicked():
    print('1')
def button2_clicked():
    print('2')
def button3_clicked():
    print('3')

app = QApplication(sys.argv)  # application 객체 생성

w = QWidget()  # default Interface 선언
w.setWindowTitle('Button')  # 위젯 제목 설정
w.setGeometry(300, 300, 300, 300)  # 위젯 크기

button1 = QPushButton('1', w)  # 버튼 추가
button1.resize(100, 100)  # 버튼 크기
button1.move(0, 100)  # 버튼 위치
button1.clicked.connect(button1_clicked)

button2 = QPushButton('2', w)  # 버튼 추가
button2.resize(100, 100)  # 버튼 크기
button2.move(100, 100)  # 버튼 위치
button2.clicked.connect(button2_clicked)

button3 = QPushButton('3', w)  # 버튼 추가
button3.resize(100, 100)  # 버튼 크기
button3.move(200, 100)  # 버튼 위치
button3.clicked.connect(button3_clicked)

w.show()  # 스크린에 띄우기

sys.exit(app.exec_())
