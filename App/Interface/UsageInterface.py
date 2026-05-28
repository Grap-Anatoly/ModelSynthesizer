import sys
import traceback
import logging
import multiprocessing
multiprocessing.freeze_support()

# Additional Classes
from App.DataProcession import Processing, TensorDataProcessing
from App.ModelsStudy import Models

# Folders name sorting
import natsort

# Interface
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QStyle, QTabWidget

# OS interaction, files in folders
import os
import pandas as pd
from os import listdir
from os.path import isfile, join


class Usage(QMainWindow):
    def __init__(self, parent=None):
        super(Usage, self).__init__(parent)

        self.title = 'Використання Моделей'
        self.width = 500
        self.height = 400
        self.initTabs()

    def initTabs(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.setFixedSize(self.width, self.height)

        self.layout = QVBoxLayout()

        self.consoleOutput = QPlainTextEdit()
        self.consoleOutput.insertPlainText("Область виведення повідомлень... \n")

        self.foldersLabel = QLabel()
        self.foldersLabel.setText("--------")
        self.model = QFileSystemModel()

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setColumnWidth(0, 250)
        self.tree.setAlternatingRowColors(True)

        if self.foldersLabel != "--------":
            self.model.setRootPath(self.foldersLabel.text())
            self.tree.setRootIndex(self.model.index(self.foldersLabel.text()))

        # self.layout.addWidget(tabs)
        self.layout.addWidget(self.identificationTab())

        self.outputlayout = QHBoxLayout()

        self.outputlayout.addWidget(self.consoleOutput)
        self.outputlayout.addWidget(self.tree)

        widget = QWidget()
        widget.setContentsMargins(-1, -1, -1, 0)
        widget.setLayout(self.layout)
        self.layout.addLayout(self.outputlayout)
        self.setCentralWidget(widget)
        self.show()

    def identificationTab(self):
        identificationTab = QWidget()
        layout = QGridLayout()

        self.modelLabel = QLineEdit()
        self.modelLabel.setPlaceholderText("Шлях до папки з моделями...")
        self.modelLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.modelLabel.setReadOnly(True)

        self.audioFilePathLabel = QLineEdit()
        self.audioFilePathLabel.setPlaceholderText("Шлях до аудіофайлу...")
        self.audioFilePathLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.audioFilePathLabel.setReadOnly(True)

        self.pBar = QProgressBar(self)
        self.pBar.setAlignment(QtCore.Qt.AlignCenter)
        self.pBar.setStyleSheet("QProgressBar"
                                  "{"
                                  "background-color : white;"
                                  "border : 1px solid grey;"
                                  "}")

        self.useModelLabel = QLabel()
        self.useModelLabel.setText("--------")
        self.useModelLabel.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showWindowInfo), 0, 0, 1, 2)

        layout.addWidget(QPushButton("Вкажіть папку з моделями", clicked=self.getModelFilePath), 1, 0)
        layout.addWidget(self.modelLabel, 1, 1)
        layout.addWidget(QPushButton("Вкажіть аудіофайл", clicked=self.getAudioFilePath), 2, 0)
        layout.addWidget(self.audioFilePathLabel, 2, 1)
        layout.addWidget(QPushButton("Розпізнавання", clicked=self.useModel), 3, 0, 1, 2)
        layout.addWidget(self.pBar, 4, 0, 1, 2)
        layout.addWidget(self.useModelLabel, 5, 0, 1, 2)

        identificationTab.setLayout(layout)

        return identificationTab

    def showWindowInfo(self):

        self.infoBox = QMessageBox()
        self.infoBox.setIcon(QMessageBox.Information)
        self.infoBox.setWindowTitle("Інформація")
        self.infoBox.setText("  Процес ідентифікації звукового повідомлення.\n"
                             "1.Вкажіть шлях до папки з "
                             "  бажаною категорією \n"
                             "  моделей нейромереж; \n"
                             "2.Оберіть аудіофайл для ідентифікації\n"
                             "3.Натисніть на кнопку 'Розпізнавання'\n"
                             "  та вкажіть порожню папку для збереження\n"
                             "  проміжних даних.")
        self.infoBox.setStandardButtons(QMessageBox.Close)

        returnValue = self.infoBox.exec()
        if returnValue == QMessageBox.Close:
            print('OK clicked')

    # Folder Tree Output

    def getOutherTree(self, path):
        # For file selection
        before, sep, after = path.rpartition("/")
        self.model.setRootPath(before + sep)
        self.tree.setRootIndex(self.model.index(before + sep))

    def getInnerTree(self, path):
        # For folder selection
        self.model.setRootPath(path + '/')
        self.tree.setRootIndex(self.model.index(path + '/'))

    # Get paths

    def getModelFilePath(self):
        file = QFileDialog.getExistingDirectory(self, "Оберіть папку з моделями")

        # return files
        if file and file != "Шлях до папки  моделями...":
            self.modelLabel.setText(file)

            self.getInnerTree(file)
            self.consoleOutput.insertPlainText("Категорію моделей обрано... \n")
        else:
            self.consoleOutput.insertPlainText("Некоректний формат файлу! \n")

    def getAudioFilePath(self):
        file, check = QFileDialog.getOpenFileName(None, "Оберіть файл",
                                                  "", "All Files (*);;Python Files (*.py);;Text Files (*.txt)")
        # return files
        if file and file != "Шлях до аудіофайлу...":

            before, sep, after = file.rpartition(".")

            if after == "wav":
                self.audioFilePathLabel.setText(file)

                self.getOutherTree(file)
                self.consoleOutput.insertPlainText("Файл обрано... \n")
            else:
                self.consoleOutput.insertPlainText("Некоректний формат файлу! \n")

    # Main Processing

    def useModel(self):

        self.consoleOutput.insertPlainText("Іде обробка... \n")

        saveFilePath = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження проміжного результату")
        filePath = saveFilePath + "/"
        modelPath = self.modelLabel.text()

        # Get all models in folder (add check?)
        modelsList = []
        for model in os.listdir(modelPath):
            modelsList.append(join(modelPath, model))

        identificationResults = []

        # Process files
        for m in modelsList:
            if self.audioFilePathLabel.text().endswith(".wav") == True:

                # File cutting

                self.consoleOutput.insertPlainText("Розбиття файлу на частини: \n")

                Processing.splitAudioFile(self.audioFilePathLabel.text(), 1000, saveFilePath, self.pBar, self.consoleOutput)

                # File to data

                self.consoleOutput.insertPlainText("Перетворення звукових файлів на числові дані: \n")

                filesList = [f for f in listdir(filePath) if
                             isfile(join(filePath, f))]
                filesList = natsort.natsorted(filesList)

                # Params

                params = open(m + '\params.txt', 'r')
                Lines = params.readlines()

                rangeStart = int(Lines[0].strip("\n"))
                rangeEnd = int(Lines[1].strip("\n")) + 1

                allyLabel = ""

                Processing.processFiles(filePath, filesList, allyLabel,  # int(self.allyLabel.text()),
                                        rangeStart, rangeEnd, filePath, self.pBar, self.consoleOutput)

                # File Deletion

                toDelete = os.listdir(saveFilePath)

                for item in toDelete:
                    if item.endswith(".wav"):
                        os.remove(os.path.join(saveFilePath, item))

            else:
                self.consoleOutput.insertPlainText("Некоректний формат файлу! \n")

            # Get ExcelFile

            excelFiles = os.listdir(saveFilePath)

            dataFile = ""

            for item in excelFiles:
                if item.endswith(".xlsx"):
                    dataFile = item

            dataFile = filePath + dataFile
            xls = pd.ExcelFile(dataFile)
            l = len(xls.sheet_names)

            self.consoleOutput.insertPlainText("Ідентифікація: \n")

            # If 2 pages
            if l > 1:

                ampData_features, ampData_labels, ampData_fileNames, phaseData_features,\
                phaseData_labels, phaseData_fileNames, headers = TensorDataProcessing.getMultipleSheetDataSet(dataFile)

                # testResultAmp = Models.useModels(modelPath, ampData_features, ampData_labels)
                testResultAmp = Models.useModels(m, ampData_features, ampData_labels)

                # resultPh = Models.useModels(modelPath, phaseData_features, phaseData_labels)
                identificationResults.append(testResultAmp)
                self.pBar.setValue(100)
                # self.useModelLabel.setText(testResultAmp)
            # If 1 page
            else:

                combData_features, combData_labels, \
                combData_fileNames, headers = TensorDataProcessing.getSingleSheetDataSet(dataFile)

                # result = Models.useModels(modelPath, combData_features, combData_labels)
                result = Models.useModels(m, combData_features, combData_labels)
                identificationResults.append(result)
                self.pBar.setValue(100)
                # self.useModelLabel.setText(result)

            # File Deletion

            xls.close()
            toDeleteExcel = os.listdir(saveFilePath)

            for item in toDeleteExcel:
                if item.endswith(".xlsx"):
                    os.remove(saveFilePath + "/" + item)

        self.audioFilePathLabel.setText("--------")

        # All results

        self.consoleOutput.insertPlainText("Всі результати моделей: \n")

        for r in identificationResults:
            self.consoleOutput.insertPlainText(f"{r} \n")

        # Best model by acc
        modelsAcc = {}
        for r in identificationResults:

            for k, v in r.items():
                b, s, a = v.rpartition(", Точність:")
                if b == 'Результат: Cвій':
                    before, sep, after = v.rpartition(", файли:")
                    b, s, a = before.rpartition(", Точність:")
                    modelsAcc[k] = a
                else:
                    self.useModelLabel.setText("Чужий")

        if len(modelsAcc) > 0:
            maxKey = max(modelsAcc, key=modelsAcc.get)
            res = f"Свій, Модель: {maxKey}; точність ідентифікації: {modelsAcc[maxKey]}"
            # res = "Свій, Модель та точність: %s" % (max(modelsAcc, key=modelsAcc.get),)

        self.useModelLabel.setText(res)
        self.consoleOutput.insertPlainText("Готово! \n")

        print(identificationResults)

# Exception catch

log = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
log.addHandler(handler)

# Exception caught

log = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
log.addHandler(handler)


def show_exception_box(log_msg):
    # Check if instance is available and show exception, else log
    if QApplication.instance() is not None:
        errorbox = QMessageBox()
        errorbox.setText("Oops. An unexpected error occured:\n{0}".format(log_msg))
        errorbox.exec_()
    else:
        log.debug("No QApplication instance available.")


class UncaughtHook(QtCore.QObject):
    _exception_caught = pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super(UncaughtHook, self).__init__(*args, **kwargs)

        sys.excepthook = self.exception_hook

        self._exception_caught.connect(show_exception_box)

    def exception_hook(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        else:
            exc_info = (exc_type, exc_value, exc_traceback)
            log_msg = '\n'.join([''.join(traceback.format_tb(exc_traceback)),
                                 '{0}: {1}'.format(exc_type.__name__, exc_value)])
            log.critical("Uncaught exception:\n {0}".format(log_msg), exc_info=exc_info)

            self._exception_caught.emit(log_msg)

qt_exception_hook = UncaughtHook()