"""

    Window for Input Data Array (Масив вхідних данних)creation

"""
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, \
    QVBoxLayout, QStyle, QTabWidget, QFileSystemModel, QLabel

from App.Interface.MVD import AudioFile, PhotoFile, VideoFile, TextFile

class MVDCreationWindow(QMainWindow):
    """
        Init method: window params, local data arrays
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.title = 'Формування МВД'
        self.width = 300
        self.height = 200
        self.initMainWindow()

    """
        Main window: choose data type
    """
    def initMainWindow(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.resize(self.width, self.height)

        self.layout = QVBoxLayout()

        self.infoLabel = QLabel()
        self.infoLabel.setText("Оберіть тип даних")
        self.infoLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.layout.addWidget(self.infoLabel)
        self.layout.addWidget(QPushButton("Аудіо-Файли", self, clicked=self.getAudioFileProcessionWindow))

        self.photoButton = QPushButton("Фотографії", self, clicked=self.getPhotoFileProcessionWindow)
        self.photoButton.setEnabled(False)
        self.layout.addWidget(self.photoButton)

        # self.layout.addWidget(QPushButton("Фотографії", self, clicked=self.getPhotoFileProcessionWindow))

        self.textButton = QPushButton("Текстові Файли", self, clicked=self.getTextFileProcessionWindow)
        self.textButton.setEnabled(False)
        self.layout.addWidget(self.textButton)

        # self.layout.addWidget(QPushButton("Текстові Файли", self, clicked=self.getTextFileProcessionWindow))

        self.videoButton = QPushButton("Відео-Файли", self, clicked=self.getVieoFileProcessionWindow)
        self.videoButton.setEnabled(False)
        self.layout.addWidget(self.videoButton)

        # self.layout.addWidget(QPushButton("Відео-Файли", self, clicked=self.getVieoFileProcessionWindow))

        widget = QWidget()
        widget.setContentsMargins(-1, -1, -1, 0)
        widget.setLayout(self.layout)

        self.setCentralWidget(widget)
        self.show()


    def getAudioFileProcessionWindow(self):
        AudioFile.FileProcessionWindow(self).show()

    def getPhotoFileProcessionWindow(self):
        PhotoFile.FileProcessionWindow(self).show()

    def getTextFileProcessionWindow(self):
        TextFile.FileProcessionWindow(self).show()

    def getVieoFileProcessionWindow(self):
        VideoFile.FileProcessionWindow(self).show()
