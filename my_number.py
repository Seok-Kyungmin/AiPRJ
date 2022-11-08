import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import tensorflow as tf

class my_number(QMainWindow):

    def __imit__(self):
        super().__init__()
        self.image = QImage(QSize(400, 400), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.brush_size = 30
        self.brush_color = Qt.black
        self.last_point = QPoint()
        self.load_model = None

        self.initUI()

    def initUI(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu("File")

        load_model_action = QAction("Load model", self)
        load_model_action.setShortcut("Ctrl+L")
        load_model_action.triggered.connect(self.load_model)

        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save)

        clear_action = QAction("Clear", self)
        clear_action.setShortcut("Ctrl+C")
        clear_action.triggered.connect(self.clear)

        filemenu.addAction(load_model_action)
        filemenu.addAction(save_action)
        filemenu.addAction(clear_action)

        self.statusbar = self.statusBar()

        self.setWindowTitle("MNISt부터 생성한 학습모델을 이용한 손글씨 인식하기")
        self.setGeometry(300, 300, 400, 400)
        self.show()

        # 그리기 이벤트
        def paintEvent(self, e):
            canvas = QPainter(self)
            canvas.drawImage(self.rect(), self.image, self.image.rect())

        def mousePressEvent(self, e):
            if e.button() == Qt.LeftButton:
                self.drawing = True
                self.last_point = e.pos()

        def mouseMoveEvent(self, e):
            if