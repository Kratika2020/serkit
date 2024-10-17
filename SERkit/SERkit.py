# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 19:57:59 2021

@author: Kratika


SERkit main application file
attachments: welserkit,serkit_gui,allmodules,speech_emotion_recognitionCode 
total 4 attachment files + 5 gif + 1 bg image
"""


import sys
import platform
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from welserkit import Ui_Form1
from serkit_gui import Ui_Form


counter = 0

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

      
class SplashScreen(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_Form1()
        self.ui.setupUi(self)

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.progress)
        self.timer.start(60)

        self.show()
       
    def progress(self):

        global counter
        self.ui.progressBar.setValue(counter)

        if counter > 100:
           
            self.timer.stop()

            self.main = MainWindow()
            self.main.show()

            self.close()

       
        counter += 1




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SplashScreen()
    sys.exit(app.exec_())