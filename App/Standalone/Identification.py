import sys
import traceback
import logging

from App.DataProcession import Processing, TensorDataProcessing
from App.ModelsStudy import Models

from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal

# OS interaction, files in folders
import os
import pandas as pd
from os import listdir
from os.path import isfile, join

# Folders name sorting
import natsort

# Tensor
import tensorflow as tf

application_path = os.path.dirname(sys.executable)

print(tf.__version__)

class Identification(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Ідентифікація')
        self.resize(270, 110)

        # Create a top-level layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        # Create the tab widget with tabs
        tabs = QTabWidget()
        tabs.addTab(self.identificationTab(), "Ідентифікація")

        layout.addWidget(tabs)

    def identificationTab(self):
        identificationTab = QWidget()
        layout = QVBoxLayout()

        self.modelLabel = QLabel()
        self.modelLabel.setText("--------")
        self.modelLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.audioFilePathLabel = QLabel()
        self.audioFilePathLabel.setText("--------")
        self.audioFilePathLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.useModelLabel = QLabel()
        self.useModelLabel.setText("--------")
        self.useModelLabel.setAlignment(QtCore.Qt.AlignCenter)


        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showWindowInfo))

        layout.addWidget(QPushButton("Вкажіть шлях до папки з моделю", clicked=self.getModelFilePath))
        layout.addWidget(self.modelLabel)
        layout.addWidget(QPushButton("Вкажіть аудіофайл", clicked=self.getAudioFilePath))
        layout.addWidget(self.audioFilePathLabel)
        layout.addWidget(QPushButton("Розпізнавання", clicked=self.useModel))
        layout.addWidget(self.useModelLabel)

        identificationTab.setLayout(layout)
        return identificationTab

    def showWindowInfo(self):

        self.infoBox = QMessageBox()
        self.infoBox.setIcon(QMessageBox.Information)
        self.infoBox.setWindowTitle("Інформація")
        self.infoBox.setText("Процес ідентифікації звукового повідомлення.\n"
                             "1.Вкажіть шлях до папки з "
                             "навчаною моделю нейромережі;\n"
                             "2.Оберіть аудіофайл для ідентифікації\n"
                             "3.Натисніть на кнопку 'Розпізнавання'\n"
                             "та вкажіть пусту папку для збереження\n"
                             "проміжних даних.")
        self.infoBox.setStandardButtons(QMessageBox.Close)

        returnValue = self.infoBox.exec()
        if returnValue == QMessageBox.Close:
            print('OK clicked')

    def getModelFilePath(self):
        file = QFileDialog.getExistingDirectory(self, "Оберіть папку")

        # return files
        if file and file != "--------":

            self.modelLabel.setText(file)
            print(file)
        else:
            self.modelLabel.setText("Некоректний формат файлу")

    def getAudioFilePath(self):
        file, check = QFileDialog.getOpenFileName(None, "Оберіть файл",
                                                  "", "All Files (*);;Python Files (*.py);;Text Files (*.txt)")
        # return files
        if file:

            before, sep, after = file.rpartition(".")

            if after == "wav":
                self.audioFilePathLabel.setText(file)
            else:
                self.audioFilePathLabel.setText("Некоректний формат файлу")

    def useModel(self):

        saveFilePath = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження проміжного результату")

        filePath = saveFilePath + "/"

        modelPath = self.modelLabel.text()

        if self.audioFilePathLabel.text().endswith(".wav") == True:

            # File cutting

            self.useModelLabel.setText("Іде обробка")

            Processing.splitAudioFile(self.audioFilePathLabel.text(), 1000, saveFilePath)

            # File to data

            filesList = [f for f in listdir(filePath) if
                         isfile(join(filePath, f))]
            filesList = natsort.natsorted(filesList)

            # Params

            params = open(modelPath + '\params.txt', 'r')
            Lines = params.readlines()

            rangeStart = int(Lines[0].strip("\n"))
            rangeEnd = int(Lines[1].strip("\n")) + 1

            allyLabel = ""

            Processing.processFiles(filePath, filesList, allyLabel,  # int(self.allyLabel.text()),
                                    rangeStart, rangeEnd, filePath)

            self.useModelLabel.setText("Файли оброблено")

            # File Deletion

            toDelete = os.listdir(saveFilePath)

            for item in toDelete:
                if item.endswith(".wav"):
                    os.remove(os.path.join(saveFilePath, item))

        else:
            self.useModelLabel.setText("Некоректний формат файлу")

        # Get ExcelFile

        excelFiles = os.listdir(saveFilePath)

        dataFile = ""

        for item in excelFiles:
            if item.endswith(".xlsx"):
                dataFile = item

        dataFile = filePath + dataFile
        xls = pd.ExcelFile(dataFile)
        l = len(xls.sheet_names)

        # If 2 pages
        if l > 1:

            self.useModelLabel.setText("Іде обробка")

            ampData_features, ampData_labels, ampData_fileNames, phaseData_features,\
            phaseData_labels, phaseData_fileNames, headers = TensorDataProcessing.getMultipleSheetDataSet(dataFile)

            testResultAmp = Models.useModels(modelPath, ampData_features, ampData_labels)

            # resultPh = Models.useModels(modelPath, phaseData_features, phaseData_labels)

            self.useModelLabel.setText(testResultAmp)
        # If 1 page
        else:

            self.useModelLabel.setText("Іде обробка")

            combData_features, combData_labels, \
            combData_fileNames, headers = TensorDataProcessing.getSingleSheetDataSet(dataFile)

            result = Models.useModels(modelPath, combData_features, combData_labels)

            self.useModelLabel.setText(result)

        # File Deletion

        xls.close()
        toDeleteExcel = os.listdir(saveFilePath)

        for item in toDeleteExcel:
            if item.endswith(".xlsx"):
                os.remove(saveFilePath + "/" + item)

        self.audioFilePathLabel.setText("--------")

# Exception caught

log = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
log.addHandler(handler)


def show_exception_box(log_msg):
    # Check if instance is available and show exception, else log
    if QApplication.instance() is not None:
        errorbox = QMessageBox()
        errorbox.setText("! Виникла неочікувана помилка:\n{0}".format(log_msg))
        errorbox.exec_()
    else:
        log.debug("Немає доступного екземпляра QApplication.")


class UncaughtHook(QtCore.QObject):
    _exception_caught = pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super(UncaughtHook, self).__init__(*args, **kwargs)

        # this registers the exception_hook() function as hook with the Python interpreter
        sys.excepthook = self.exception_hook

        # connect signal to execute the message box function always on main thread
        self._exception_caught.connect(show_exception_box)

    def exception_hook(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # ignore keyboard interrupt to support console applications
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        else:
            exc_info = (exc_type, exc_value, exc_traceback)
            log_msg = '\n'.join([''.join(traceback.format_tb(exc_traceback)),
                                 '{0}: {1}'.format(exc_type.__name__, exc_value)])
            log.critical("Неочікуваний виняток:\n {0}".format(log_msg), exc_info=exc_info)

            # trigger message box show
            self._exception_caught.emit(log_msg)


# create a global instance of our class to register the hook
qt_exception_hook = UncaughtHook()

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Identification()
    ex.show()

    sys.exit(app.exec_())