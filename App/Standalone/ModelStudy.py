import sys
import traceback
import logging

from App.DataProcession import Processing, TensorDataProcessing
import multiprocessing
multiprocessing.freeze_support()
from App.ModelsStudy import Models

from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal

# OS interaction, files in folders
import os
import shutil
import pandas as pd
from os import listdir
from os.path import isfile, join

# Folders name sorting
import natsort

# Processing
import joblib
from joblib import Parallel, delayed

application_path = os.path.dirname(sys.executable)

# Interface -----------------------------------------------------------------------------------------------------------

class Main(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Аудіо')
        self.resize(270, 110)

        # Create a top-level layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        # Create the tab widget with tabs

        tabs = QTabWidget()
        tabs.addTab(self.initialFileProcessionTab(), "Розбиття початкового файлу")
        tabs.addTab(self.numericalDataProcessionTab(), "Отримання числових даних")
        tabs.addTab(self.dataSetProcessionTab(), "Формування первинного опису")
        tabs.addTab(self.neuralModelsTab(), "Робота з моделями")
        layout.addWidget(tabs)

    def initialFileProcessionTab(self):
        generalTab = QWidget()
        layout = QVBoxLayout()

        self.filepathLabel = QLabel()
        self.filepathLabel.setText("--------")
        self.filepathLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.chunkInfo = QLabel(self)
        self.chunkInfo.setText("Введіть довжину відрізку, мс")
        self.chunkInfo.setAlignment(QtCore.Qt.AlignCenter)

        self.textbox = QLineEdit(self)
        self.textbox.setText("1000")

        self.chunkLabel = QLabel()
        self.chunkLabel.setText("--------")
        self.chunkLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.cuttingLabel = QLabel()
        self.cuttingLabel.setText("--------")
        self.cuttingLabel.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showFirstWindowInfo))
        layout.addWidget(QPushButton("Виберіть файл для розбиття", clicked=self.getInitialFileFolder))
        layout.addWidget(self.filepathLabel)

        layout.addWidget(self.chunkInfo)
        layout.addWidget(self.textbox)
        layout.addWidget(QPushButton('Ввести', clicked=self.enterChunkLength))
        layout.addWidget(self.chunkLabel)

        layout.addWidget(QPushButton("Розбити файл", clicked=self.processFile))
        layout.addWidget(self.cuttingLabel)

        generalTab.setLayout(layout)
        return generalTab

    def numericalDataProcessionTab(self):
        audioTab = QWidget()
        layout = QVBoxLayout()

        self.folderPathLabel = QLabel()
        self.folderPathLabel.setText("--------")
        self.folderPathLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.rangeInfo = QLabel(self)
        self.rangeInfo.setText("Введіть діапазон виведення даних, через кому, Гц")
        self.rangeInfo.setAlignment(QtCore.Qt.AlignCenter)

        self.rangeStartTextbox = QLineEdit(self)
        self.rangeStartTextbox.setText("16")

        self.rangeEndTextbox = QLineEdit(self)
        self.rangeEndTextbox.setText("350")

        self.rangeLabel = QLabel()
        self.rangeLabel.setText("--------")
        self.rangeLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.allyInfo = QLabel(self)
        self.allyInfo.setText("Введіть 'свій або чужий', 100/0 ")
        self.allyInfo.setAlignment(QtCore.Qt.AlignCenter)

        self.allyTextbox = QLineEdit(self)
        self.allyTextbox.setText("100")

        self.allyLabel = QLabel()
        self.allyLabel.setText("--------")
        self.allyLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.processingLabel = QLabel()
        self.processingLabel.setText("--------")
        self.processingLabel.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showSecondWindowInfo))
        layout.addWidget(QPushButton("Вкажіть шлях до файлів для аналізу", clicked=self.getFilesFolderPath))
        layout.addWidget(self.folderPathLabel)

        layout.addWidget(self.rangeInfo)
        layout.addWidget(self.rangeStartTextbox)
        layout.addWidget(self.rangeEndTextbox)
        layout.addWidget(QPushButton('Ввести', clicked=self.enterRange))
        layout.addWidget(self.rangeLabel)

        layout.addWidget(self.allyInfo)
        layout.addWidget(self.allyTextbox)
        layout.addWidget(QPushButton('Ввести', clicked=self.enterPointer))
        layout.addWidget(self.allyLabel)

        layout.addWidget(QPushButton("Перетворити", clicked=self.processAudioFiles))
        layout.addWidget(self.processingLabel)

        audioTab.setLayout(layout)
        return audioTab

    def dataSetProcessionTab(self):
        excelTab = QWidget()
        layout = QVBoxLayout()

        self.dataFolderPathLabel = QLabel()
        self.dataFolderPathLabel.setText("--------")
        self.dataFolderPathLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.fileNameInfo = QLabel(self)
        self.fileNameInfo.setText("Вкажіть назву нового файлу")
        self.fileNameInfo.setAlignment(QtCore.Qt.AlignCenter)

        self.fileNameBox = QLineEdit(self)
        self.fileNameBox.setText("Назва Файлу")

        self.processingDataSetLabel = QLabel()
        self.processingDataSetLabel.setText("--------")
        self.processingDataSetLabel.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showThirdWindowInfo))
        layout.addWidget(QPushButton("Вкажіть шлях до файлів для об`єднання", clicked=self.getExcelFilesFolderPath))
        layout.addWidget(self.dataFolderPathLabel)

        layout.addWidget(self.fileNameInfo)
        layout.addWidget(self.fileNameBox)

        layout.addWidget(
            QPushButton("Сформувати первинний опис (Амп і фаза окремо)", clicked=self.createDataSet))
        layout.addWidget(
            QPushButton("Сформувати первинний опис (Амп і фаза разом)", clicked=self.createCombinedDataSet))
        layout.addWidget(self.processingDataSetLabel)

        excelTab.setLayout(layout)
        return excelTab

    def neuralModelsTab(self):
        modelTab = QWidget()
        layout = QVBoxLayout()

        self.rawDataPathLabel = QLabel()
        self.rawDataPathLabel.setText("--------")
        self.rawDataPathLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.usageDataPathLabel = QLabel()
        self.usageDataPathLabel.setText("--------")
        self.usageDataPathLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.firstModelLabel = QLabel()
        self.firstModelLabel.setText("--------")
        self.firstModelLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.modelPathLabel = QLabel()
        self.modelPathLabel.setText("--------")
        self.modelPathLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.identificationDataPathLabel = QLabel()
        self.identificationDataPathLabel.setText("--------")
        self.identificationDataPathLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.useModelLabel = QLabel()
        self.useModelLabel.setText("--------")
        self.useModelLabel.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showFourthWindowInfo))

        layout.addWidget(QPushButton("Вкажіть шлях до файлу з даними", clicked=self.getDataFilePath))
        layout.addWidget(self.rawDataPathLabel)
        layout.addWidget(QPushButton("Вкажіть шлях до файлу з тестовими даними", clicked=self.getTestDataFilePath))
        layout.addWidget(self.usageDataPathLabel)

        layout.addWidget(QPushButton("Навчити набір моделей", clicked=self.studyModels))
        layout.addWidget(self.firstModelLabel)

        layout.addWidget(QPushButton("Вкажіть шлях до папки з моделю", clicked=self.getModelFilePath))
        layout.addWidget(self.modelPathLabel)
        layout.addWidget(QPushButton("Оберіть набір даних для розпізнавання", clicked=self.getIdentificationDataFilePath))
        layout.addWidget(self.identificationDataPathLabel)
        layout.addWidget(QPushButton("Ідентифікація", clicked=self.useModel))
        layout.addWidget(self.useModelLabel)

        modelTab.setLayout(layout)
        return modelTab

    # Info Text -----------------------------------------------------------------------------------------------------

    def showFirstWindowInfo(self):

        self.infoBox = QMessageBox()
        self.infoBox.setIcon(QMessageBox.Information)
        self.infoBox.setWindowTitle("Інформація")
        self.infoBox.setText("Процес розбиття аудіофайлу на частини.\n"
                             "1.Вкажіть аудіофайл для подальшого розбиття;\n"
                             "2.Вкажіть бажану довжину нових файлів в мілісекундах;\n"
                             "1000 мс = 1 секунда\n"
                             "(Звертайте увагу на довжину початкового файлу\n"
                             "і обирайте відповідну довжину відрізків)\n"
                             "3.Оберіть місце для збереження нових файлів і\n"
                             "натисніть на кнопку 'Розбити файл';\n"
                             "(Для комфортної подальшої роботи\n"
                             "рекомендується створити окрему папку для файлів)")
        self.infoBox.setStandardButtons(QMessageBox.Close)

        returnValue = self.infoBox.exec()

    def showSecondWindowInfo(self):

        self.infoBox = QMessageBox()
        self.infoBox.setIcon(QMessageBox.Information)
        self.infoBox.setWindowTitle("Інформація")
        self.infoBox.setText("Процес перетворення аудіофайлу в числові дані.\n"
                             "1.Вкажіть шлях до аудіофайлів для перетворення;\n"
                             "2.Вкажіть частотний діапазон\n"
                             "в якому будуть виведені числові дані\n"
                             "(Діапазон людського слуху 16 - 20000 гц);\n"
                             "3.Додайте вказівник свій/чужий,\n"
                             "де свій це 100 а чужий, 0;\n"
                             "4.Оберіть місце для збереження файлу і\n"
                             "натисніть на кнопку 'Перетворити'.\n"
                             "(Для комфортної подальшої роботи\n"
                             "рекомендується створити окрему папку для файлів)")
        self.infoBox.setStandardButtons(QMessageBox.Close)

        returnValue = self.infoBox.exec()

    def showThirdWindowInfo(self):

        self.infoBox = QMessageBox()
        self.infoBox.setIcon(QMessageBox.Information)
        self.infoBox.setWindowTitle("Інформація")
        self.infoBox.setText("Процес створення первинного опису.\n"
                             "1.Вкажіть шлях до бажаних файлів для об`єднання;\n"
                             "(Якщо це не було зроблено раніше,\n"
                             "створіть окрему папку та перемістіть файли туди)\n"
                             "2.Вкажіть бажану назву нового файлу;\n"
                             "3.Оберіть тип нового файлу (одна з двох кнопок)\n"
                             "і місце для збереження результату.")
        self.infoBox.setStandardButtons(QMessageBox.Close)

        returnValue = self.infoBox.exec()

    def showFourthWindowInfo(self):

        self.infoBox = QMessageBox()
        self.infoBox.setIcon(QMessageBox.Information)
        self.infoBox.setWindowTitle("Інформація")
        self.infoBox.setText("Процес роботи з моделями.\n"
                             "Процес поділений на дві частини\n"
                             "Навчання та Робота.\n"
                             "Навчання:\n"
                             "1.Вкажіть шлях до бажаного файлу з даними;\n"
                             "(Первинний опис, створений раніше)\n"
                             "2.Вкажіть шлях до  файлу з тестовими даними;\n"
                             "3.Вкажіть місце для збереження навчаних моделей\n"
                             "і натисніть на кнопку;\n"
                             "(Для збереження моделей\n"
                             "рекомендується створити окрему папку)\n"
                             "Робота:\n"
                             "1.Вкажіть шлях до папки з бажаною моделю;\n"
                             "(Необхідно обрати папку з усіма файлами моделі)\n"
                             "2.Вкажіть шлях до бажаного файлу з даними для аналізу;\n"
                             "3.Натисніть на кнопку 'Використати модель'.")
        self.infoBox.setStandardButtons(QMessageBox.Close)

        returnValue = self.infoBox.exec()

    # Paths set -----------------------------------------------------------------------------------------------------

    def getInitialFileFolder(self):
        file, check = QFileDialog.getOpenFileName(None, "Оберіть файл",
                                                  "", "All Files (*);;Python Files (*.py);;Text Files (*.txt)")
        # return files
        if  check:
            print(file)
            self.filepathLabel.setText(file)

    def getFilesFolderPath(self):
        file = QFileDialog.getExistingDirectory(self, "Оберіть папку")

        # return files
        if file:
            print(file)
            self.folderPathLabel.setText(file + '/')

    def getExcelFilesFolderPath(self):
        file = QFileDialog.getExistingDirectory(self, "Оберіть папку")

        # return files
        if file:
            print(file)
            self.dataFolderPathLabel.setText(file + '/')

    def getDataFilePath(self):
        file, check = QFileDialog.getOpenFileName(None, "Оберіть файл",
                                           "", "All Files (*);;Python Files (*.py);;Text Files (*.txt)")
        # return files
        if file:

            before, sep, after = file.rpartition(".")

            if after == "xlsx":
                self.rawDataPathLabel.setText(file)
            else:
                self.rawDataPathLabel.setText("Некоректний формат файлу")

    def getTestDataFilePath(self):
        file, check = QFileDialog.getOpenFileName(None, "Оберіть файл",
                                           "", "All Files (*);;Python Files (*.py);;Text Files (*.txt)")
        # return files
        if file:

            before, sep, after = file.rpartition(".")

            if after == "xlsx":
                self.usageDataPathLabel.setText(file)
            else:
                self.usageDataPathLabel.setText("Некоректний формат файлу")

    def getModelFilePath(self):
        file = QFileDialog.getExistingDirectory(self, "Оберіть папку")

        # return files
        if file and file != "--------":

            self.modelPathLabel.setText(file)
        else:
            self.modelPathLabel.setText("Некоректний формат файлу")

    def getIdentificationDataFilePath(self):
        file, check = QFileDialog.getOpenFileName(None, "Оберіть файл",
                                           "", "All Files (*);;Python Files (*.py);;Text Files (*.txt)")
        # return files
        if file:

            before, sep, after = file.rpartition(".")

            if after == "xlsx" or after == "wav":
                self.identificationDataPathLabel.setText(file)
            else:
                self.identificationDataPathLabel.setText("Некоректний формат файлу")

    # Additional data -----------------------------------------------------------------------------------------------

    def enterChunkLength(self):

        textboxValue = self.textbox.text()

        if int(textboxValue) > 0:
            self.chunkLabel.setText(textboxValue)
        else:
            self.chunkLabel.setText("Довжина відрізку не може бути меншою за 0мс")

    def enterRange(self):

        textboxStartValue = self.rangeStartTextbox.text()
        textboxEndValue = self.rangeEndTextbox.text()

        if int(textboxStartValue) > 0 and int(textboxEndValue) > 0:
            if int(textboxStartValue) > int(textboxEndValue):
                self.rangeLabel.setText("Початок діапазону не може бути більшим за кінець")
            else:
                self.rangeLabel.setText(textboxStartValue + " / " + textboxEndValue)
        else:
            self.rangeLabel.setText("Початок і кінець діапазону не можуть бути меншими за 0")

    def enterPointer(self):
        textboxValue = self.allyTextbox.text()

        if int(textboxValue) > -1 and int(textboxValue) < 101:
            self.allyLabel.setText(textboxValue)
        else:
            self.allyLabel.setText("Вказівник не повинен виходити за встановлені межі")

    # Final processing -----------------------------------------------------------------------------------------------

    def processFile(self):
        filePath = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження результату")

        if filePath:

            before, sep, after = self.filepathLabel.text().rpartition(".")

            if after == "wav":
                self.cuttingLabel.setText("Іде обробка")

                Processing.splitAudioFile(self.filepathLabel.text(), int(self.chunkLabel.text()), filePath)
                self.cuttingLabel.setText("Файл поділено на частини")

                self.filepathLabel.setText("--------")
                self.chunkLabel.setText("--------")
            else:
                self.cuttingLabel.setText("Некоректний формат файлу")

    def processAudioFiles(self):
        filePath = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження результату")

        if filePath:

            filePath = filePath + "/"

            if self.folderPathLabel.text() != "--------":

                self.processingLabel.setText("Іде обробка")

                filesList = [f for f in listdir(self.folderPathLabel.text()) if isfile(join(self.folderPathLabel.text(), f))]
                filesList = natsort.natsorted(filesList)

                rangeStart, rangeEnd = self.rangeLabel.text().split(" / ")

                Processing.processFiles(self.folderPathLabel.text(), filesList, int(self.allyLabel.text()), int(rangeStart), int(rangeEnd), filePath)

                self.processingLabel.setText("Файли оброблено")

                self.folderPathLabel.setText("--------")
                self.rangeLabel.setText("--------")
                self.allyLabel.setText("--------")
            else:
                self.processingLabel.setText("Некоректний формат файлів для обробки")


    def createDataSet(self):
        filePath = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження результату")

        if self.dataFolderPathLabel.text() != "--------":

            self.processingDataSetLabel.setText("Іде обробка")

            filePath = filePath + "/"

            TensorDataProcessing.createDataSet(self.dataFolderPathLabel.text(), self.fileNameBox.text(), filePath)

            self.processingDataSetLabel.setText("Набір даних створено!")

            self.dataFolderPathLabel.setText("--------")
        else:
            self.processingDataSetLabel.setText("Некоректний формат файлів для обробки!")

    def createCombinedDataSet(self):
        filePath = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження результату")

        if self.dataFolderPathLabel.text() != "--------":

            self.processingDataSetLabel.setText("Іде обробка")

            filePath = filePath + "/"

            TensorDataProcessing.createCombinedDataSet(self.dataFolderPathLabel.text(), self.fileNameBox.text(), filePath)

            self.processingDataSetLabel.setText("Набір даних створено!")

            self.dataFolderPathLabel.setText("--------")
        else:
            self.processingDataSetLabel.setText("Некоректний формат файлів для обробки!")

    def studyModels(self):
        saveFilePath = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження результату")
        filePath = self.rawDataPathLabel.text()
        testFilePath = self.usageDataPathLabel.text()

        if filePath and filePath != "--------":

            xls = pd.ExcelFile(filePath)
            l = len(xls.sheet_names)

            if l > 1:
                self.firstModelLabel.setText("Іде обробка")

                ampData_features, ampData_labels, ampData_fileNames, phaseData_features, phaseData_labels, \
                phaseData_fileNames, headers = TensorDataProcessing.getMultipleSheetDataSet(filePath)

                # Test Features
                ampData_testFeatures, ampData_testLabels, ampData_fileNames, phaseData_testFeatures, \
                phaseData_testLabels, phaseData_fileNames, test_headers = TensorDataProcessing.getMultipleSheetDataSet(testFilePath)

                res = []

                def firstTypeStudy():

                    print("Process 1 starting")
                    count, resultSet = Models.studyFirstTypeModels(ampData_features, ampData_labels,
                                                                   ampData_testFeatures,
                                                                   ampData_testLabels, ampData_fileNames, saveFilePath,
                                                                   headers)


                    for i in range(len(count)):
                        # print("%sGuess accuracy: %s; " % (count[i], resultSet[i]))
                        res.append("%sFirst Model Guess accuracy: %s; " % (count[i], resultSet[i]))

                def secondTypeStudy():

                    print("Process 2 starting")
                    count, resultSet = Models.studySecondTypeModels(ampData_features, ampData_labels,
                                                                    ampData_testFeatures,
                                                                    ampData_testLabels, ampData_fileNames, saveFilePath,
                                                                    headers)

                    for i in range(len(count)):
                        # print("%sGuess accuracy: %s; " % (count[i], resultSet[i]))
                        res.append("%sSecond Model Guess accuracy: %s; " % (count[i], resultSet[i]))

                def thirdTypeStudy():

                    print("Process 3 starting")
                    count, resultSet = Models.studyThirdTypeModels(ampData_features, ampData_labels,
                                                                   ampData_testFeatures,
                                                                   ampData_testLabels, ampData_fileNames, saveFilePath,
                                                                   headers)

                    for i in range(len(count)):
                        # print("%sGuess accuracy: %s; " % (count[i], resultSet[i]))
                        res.append("%sThird Model Guess accuracy: %s; " % (count[i], resultSet[i]))

                def fourthTypeStudy():

                    print("Process 4 starting")
                    count, resultSet = Models.studyFourthTypeModels(ampData_features, ampData_labels,
                                                                    ampData_testFeatures,
                                                                    ampData_testLabels, ampData_fileNames, saveFilePath,
                                                                    headers)

                    for i in range(len(count)):
                        # print("%sGuess accuracy: %s; " % (count[i], resultSet[i]))
                        res.append("%sFourth Model Guess accuracy: %s; " % (count[i], resultSet[i]))


                # Models.studyGeneticTypeModels(ampData_features, ampData_labels,
                #                               ampData_testFeatures,
                #                               ampData_testLabels, ampData_fileNames, saveFilePath,
                #                               headers)

                number_of_cpu = joblib.cpu_count()

                delayed_funcs = []

                delayed_funcs.append((delayed(firstTypeStudy)()))
                delayed_funcs.append((delayed(secondTypeStudy)()))

                parallel_pool = Parallel(n_jobs=number_of_cpu, prefer="threads")

                parallel_pool(delayed_funcs)

                delayed_funcs_2 = []

                delayed_funcs_2.append((delayed(thirdTypeStudy)()))
                delayed_funcs_2.append((delayed(fourthTypeStudy)()))

                parallel_pool_2 = Parallel(n_jobs=number_of_cpu, prefer="threads")

                parallel_pool_2(delayed_funcs_2)

                # Model sorting

                models = os.listdir(saveFilePath)

                files = {}

                for model in models:
                    before, sep, after = model.rpartition(", ")
                    files[before] = after
                    for key, value in files.items():
                        if after > value and key != before:
                            files[before] = after

                bestModels = []
                for key, value in files.items():
                    bestModels.append(key + ", " + value)

                for filename in os.listdir(saveFilePath):
                    if filename not in bestModels:
                        shutil.rmtree(saveFilePath + "/" + filename, ignore_errors=False, onerror=None)
            else:
                self.firstModelLabel.setText("Іде обробка")

                combData_features, combData_labels, combData_fileNames, headers = TensorDataProcessing.getSingleSheetDataSet(filePath)

                # Test Features
                combData_testFeatures, combData_testLabels, combData_fileNames, test_headers = TensorDataProcessing.getSingleSheetDataSet(testFilePath)


                # count, resultSet = Models.studyFirstTypeModels(combData_features, combData_labels, combData_testFeatures,
                #                                                combData_testLabels, combData_fileNames, saveFilePath, headers)
                #
                # for i in range(len(count)):
                #     print("%sGuess accuracy: %s; " % (count[i], resultSet[i]))

                res = []

                def firstTypeStudy():

                    print("Process 1 starting")
                    count, resultSet = Models.studyFirstTypeModels(combData_features, combData_labels, combData_testFeatures,
                                                                   combData_testLabels, combData_fileNames, saveFilePath, headers)

                    for i in range(len(count)):
                        # print("%sGuess accuracy: %s; " % (count[i], resultSet[i]))
                        res.append("%sFirst Model Guess accuracy: %s; " % (count[i], resultSet[i]))

                def secondTypeStudy():

                    print("Process 2 starting")
                    count, resultSet = Models.studySecondTypeModels(combData_features, combData_labels, combData_testFeatures,
                                                                    combData_testLabels, combData_fileNames, saveFilePath, headers)

                    for i in range(len(count)):
                        # print("%sGuess accuracy: %s; " % (count[i], resultSet[i]))
                        res.append("%sSecond Model Guess accuracy: %s; " % (count[i], resultSet[i]))

                def thirdTypeStudy():

                    print("Process 3 starting")
                    count, resultSet = Models.studyThirdTypeModels(combData_features, combData_labels, combData_testFeatures,
                                                                   combData_testLabels, combData_fileNames, saveFilePath, headers)

                    for i in range(len(count)):
                        # print("%sGuess accuracy: %s; " % (count[i], resultSet[i]))
                        res.append("%sThird Model Guess accuracy: %s; " % (count[i], resultSet[i]))

                def fourthTypeStudy():

                    print("Process 4 starting")
                    count, resultSet = Models.studyFourthTypeModels(combData_features, combData_labels, combData_testFeatures,
                                                                    combData_testLabels, combData_fileNames, saveFilePath, headers)

                    for i in range(len(count)):
                        # print("%sGuess accuracy: %s; " % (count[i], resultSet[i]))
                        res.append("%sFourth Model Guess accuracy: %s; " % (count[i], resultSet[i]))

                number_of_cpu = joblib.cpu_count()

                delayed_funcs = []

                delayed_funcs.append((delayed(firstTypeStudy)()))
                delayed_funcs.append((delayed(secondTypeStudy)()))

                parallel_pool = Parallel(n_jobs=number_of_cpu, prefer="threads")

                parallel_pool(delayed_funcs)

                delayed_funcs_2 = []

                delayed_funcs_2.append((delayed(thirdTypeStudy)()))
                delayed_funcs_2.append((delayed(fourthTypeStudy)()))

                parallel_pool_2 = Parallel(n_jobs=number_of_cpu, prefer="threads")

                parallel_pool_2(delayed_funcs_2)

                # Model sorting

                models = os.listdir(saveFilePath)

                files = {}

                for model in models:
                    before, sep, after = model.rpartition(", ")
                    files[before] = after
                    for key, value in files.items():
                        if after > value and key != before:
                            files[before] = after

                bestModels = []
                for key, value in files.items():
                    bestModels.append(key + ", " + value)

                for filename in os.listdir(saveFilePath):
                    if filename not in bestModels:
                        shutil.rmtree(saveFilePath + "/" + filename, ignore_errors=False, onerror=None)

            self.firstModelLabel.setText("Моделі навчано")
            self.rawDataPathLabel.setText("--------")
            self.usageDataPathLabel.setText("--------")
        else:
            self.firstModelLabel.setText('Оберіть Файл')

    def useModel(self):

        modelPath = self.modelPathLabel.text()
        identificationDataPath = self.identificationDataPathLabel.text()

        if identificationDataPath and identificationDataPath != "--------":

            xls = pd.ExcelFile(identificationDataPath)
            l = len(xls.sheet_names)

            if l > 1:

                self.useModelLabel.setText("Іде обробка")

                ampData_features, ampData_labels, ampData_fileNames, phaseData_features, phaseData_labels, \
                phaseData_fileNames, headers = TensorDataProcessing.getMultipleSheetDataSet(identificationDataPath)

                testResultAmp = Models.useModels(modelPath, ampData_features, ampData_labels)

                # testResultPh = Models.useModels(modelPath, phaseData_features, phaseData_labels)

                self.useModelLabel.setText(testResultAmp)

                # self.testModelLabel.setText(testResultPh)
            else:

                self.useModelLabel.setText("Іде обробка")

                combData_features, combData_labels, \
                combData_fileNames, headers = TensorDataProcessing.getSingleSheetDataSet(identificationDataPath)

                testResult = Models.useModels(modelPath, combData_features, combData_labels)

                self.useModelLabel.setText(testResult)

            self.usageDataPathLabel.setText("--------")
        else:
            self.useModelLabel.setText('Оберіть Файл')

# Exception caught

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

if __name__ == '__main__':

    app = QApplication(sys.argv)

    ex = Main()
    ex.show()

    sys.exit(app.exec_())

