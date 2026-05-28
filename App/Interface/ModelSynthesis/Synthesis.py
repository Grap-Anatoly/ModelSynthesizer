"""

    Window for Model synthesis operations (Синтез моделей)

"""
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, \
    QVBoxLayout, QStyle, QTabWidget, QFileSystemModel, QLabel

# Additional Classes
from App.DataProcession import Processing, TensorDataProcessing
from App.ModelsStudy import Models

from App.Interface.ModelSynthesis import NeuralModels, MGUA

from App.Interface import ExcelDataImport

# OS interaction, files in folders
import os
import pandas as pd
import shutil

class ModelSynthesisWindow(QMainWindow):
    """
        Init method: window params, local data arrays
    """
    def __init__(self, parent=None):
        super(ModelSynthesisWindow, self).__init__(parent)

        self.title = 'Синтез моделей'
        self.width = 300
        self.height = 200
        self.initMainWindow()
    """
        Main window
    """
    def initMainWindow(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.resize(self.width, self.height)

        self.layout = QVBoxLayout()

        self.infoLabel = QLabel()
        self.infoLabel.setText("Оберіть тип моделей")
        self.infoLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.layout.addWidget(self.infoLabel)
        self.layout.addWidget(QPushButton("Нейромережі", self, clicked=self.getNeuroModelWindow))

        self.MGUAButton = QPushButton("МГУА", self, clicked=self.getMGUAWindow)
        self.MGUAButton.setEnabled(False)
        self.layout.addWidget(self.MGUAButton)

        # self.layout.addWidget(QPushButton("МГУА", self, clicked=self.getMGUAWindow))

        widget = QWidget()
        widget.setContentsMargins(-1, -1, -1, 0)
        widget.setLayout(self.layout)

        self.setCentralWidget(widget)
        self.show()

    def getNeuroModelWindow(self):
        NeuralModels.AISynthesisWindow(self).show()

    def getMGUAWindow(self):
        MGUA.MGUASynthesisWindow(self).show()

