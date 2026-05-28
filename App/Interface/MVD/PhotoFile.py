"""

    Window for Photo Data Set Creation

"""
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, \
    QVBoxLayout, QStyle, QTabWidget, QFileSystemModel, QLabel

# Additional Classes
from App.DataProcession import Processing, TensorDataProcessing

# Folders name sorting
import natsort

# OS interaction, files in folders
from os import listdir
from os.path import isfile, join

class FileProcessionWindow(QMainWindow):

    def __init__(self, parent=None):
        super(FileProcessionWindow, self).__init__(parent)

        self.title = 'Формування МВД Фото-Файлів'
        self.width = 500
        self.height = 600
        self.initFileProcessionTabs()

    def initFileProcessionTabs(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.setFixedSize(self.width, self.height)

        self.layout = QVBoxLayout()

        widget = QWidget()
        widget.setContentsMargins(-1, -1, -1, 0)
        widget.setLayout(self.layout)

        self.setCentralWidget(widget)
        self.show()