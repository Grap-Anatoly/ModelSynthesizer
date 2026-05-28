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
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, \
    QVBoxLayout, QStyle, QTabWidget, QFileSystemModel, QLabel

# OS interaction, files in folders
import os
import pandas as pd
from os import listdir
from os.path import isfile, join
import shutil


class Accumulation(QMainWindow):

    def __init__(self, parent=None):
        super(Accumulation, self).__init__(parent)

        self.title = 'Накопичення Моделей'
        self.width = 500
        self.height = 600
        self.initTabs()

    def dataReady(self):
        cursor = self.output.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(str(self.process.readAll()))
        self.output.ensureCursorVisible()

    def initTabs(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.setFixedSize(self.width, self.height)

        self.layout = QVBoxLayout()

        tabs = QTabWidget()
        tabs.addTab(self.initialFileProcessionTab(), "Розбиття початкового файлу")
        tabs.addTab(self.numericalDataProcessionTab(), "Отримання числових даних")
        tabs.addTab(self.dataSetProcessionTab(), "Формування первинного опису")
        tabs.addTab(self.neuralModelsTab(), "Робота з моделями")
        tabs.addTab(self.modelManagementTab(), "Менеджмент моделей")

        self.consoleOutput = QPlainTextEdit()
        self.consoleOutput.insertPlainText("Область виведення повідомлень... \n")
        self.consoleOutput.verticalScrollBar().setValue(self.consoleOutput.verticalScrollBar().minimum())


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


        self.layout.addWidget(tabs)

        self.outputlayout = QHBoxLayout()

        self.outputlayout.addWidget(self.consoleOutput)
        self.outputlayout.addWidget(self.tree)

        widget = QWidget()
        widget.setContentsMargins(-1, -1, -1, 0)
        widget.setLayout(self.layout)
        self.layout.addLayout(self.outputlayout)
        self.setCentralWidget(widget)
        self.show()

    def initialFileProcessionTab(self):
        generalTab = QWidget()
        layout = QGridLayout()

        self.filepathLabel = QLineEdit()
        self.filepathLabel.setPlaceholderText("Шлях до аудіофайлу...")
        self.filepathLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.filepathLabel.setReadOnly(True)

        self.chunkInfo = QLabel(self)
        self.chunkInfo.setText("Введіть довжину відрізку, мс")
        self.chunkInfo.setAlignment(QtCore.Qt.AlignCenter)

        self.chunkLengthBox = QLineEdit(self)
        self.chunkLengthBox.setText("1000")
        self.chunkLengthBox.setAlignment(QtCore.Qt.AlignLeft)

        self.chunkLabel = QLabel()
        self.chunkLabel.setText("--------")
        self.chunkLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.initialPBar = QProgressBar(self)
        self.initialPBar.setAlignment(QtCore.Qt.AlignCenter)
        self.initialPBar.setStyleSheet("QProgressBar"
                                       "{"
                                       "background-color : white;"
                                       "border : 1px solid grey;"
                                       "}")

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showFirstWindowInfo), 0, 0, 1, 2)
        layout.addWidget(QPushButton("Обрати файл", clicked=self.getInitialFileFolder), 1, 0)
        layout.addWidget(self.filepathLabel, 1, 1)

        layout.addWidget(self.chunkInfo, 2, 0, 1, 2)
        layout.addWidget(QPushButton("Ввести", clicked=self.enterChunkLength), 3, 0)
        layout.addWidget(self.chunkLengthBox, 3, 1)

        layout.addWidget(QPushButton("Розбити файл", clicked=self.processFile), 4, 0, 1, 2)

        layout.addWidget(self.initialPBar, 5, 0, 1, 2)

        generalTab.setLayout(layout)

        return generalTab

    def numericalDataProcessionTab(self):
        audioTab = QWidget()
        layout = QGridLayout()

        self.folderPathLabel = QLineEdit(self)
        self.folderPathLabel.setPlaceholderText("Шлях до папки з звуковими файлами...")
        self.folderPathLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.folderPathLabel.setReadOnly(True)

        self.rangeInfo = QLabel(self)
        self.rangeInfo.setText("Введіть діапазон виведення даних, через кому, Гц")
        self.rangeInfo.setAlignment(QtCore.Qt.AlignCenter)

        self.rangeStartTextbox = QLineEdit(self)
        self.rangeStartTextbox.setText("16")

        self.rangeEndTextbox = QLineEdit(self)
        self.rangeEndTextbox.setText("350")

        self.rangeLabel = QLabel()
        self.rangeLabel.setText("--------")

        self.allyInfo = QLabel(self)
        self.allyInfo.setText("Введіть 'свій або чужий', 100/0 ")
        self.allyInfo.setAlignment(QtCore.Qt.AlignCenter)

        self.allyTextbox = QLineEdit(self)
        self.allyTextbox.setText("100")
        self.allyTextbox.setAlignment(QtCore.Qt.AlignLeft)

        self.allyLabel = QLabel()
        self.allyLabel.setText("--------")

        self.numericalPBar = QProgressBar(self)
        self.numericalPBar.setAlignment(QtCore.Qt.AlignCenter)
        self.numericalPBar.setStyleSheet("QProgressBar"
                                       "{"
                                       "background-color : white;"
                                       "border : 1px solid grey;"
                                       "}")

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showSecondWindowInfo), 0, 0, 1, 3)
        layout.addWidget(QPushButton("Обрати папку", clicked=self.getFilesFolderPath), 1, 0)
        layout.addWidget(self.folderPathLabel, 1, 1, 1, 2)

        layout.addWidget(self.rangeInfo, 2, 0, 1, 3)
        layout.addWidget(QPushButton('Ввести', clicked=self.enterRange), 3, 0)
        layout.addWidget(self.rangeStartTextbox, 3, 1)
        layout.addWidget(self.rangeEndTextbox, 3, 2)

        layout.addWidget(self.allyInfo, 4, 0, 1, 3)
        layout.addWidget(QPushButton('Ввести', clicked=self.enterPointer), 5, 0)
        layout.addWidget(self.allyTextbox, 5, 1, 1, 2)

        # layout.addWidget(QPushButton("Перетворити", clicked=self.longRunProcessFiles), 6, 0, 1, 3)
        layout.addWidget(QPushButton("Перетворити", clicked=self.processAudioFiles), 6, 0, 1, 3)
        layout.addWidget(self.numericalPBar, 7, 0, 1, 3)

        audioTab.setLayout(layout)
        return audioTab

    def dataSetProcessionTab(self):
        excelTab = QWidget()
        layout = QGridLayout()

        self.dataFolderPathLabel = QLineEdit()
        self.dataFolderPathLabel.setPlaceholderText("Шлях до файлів для об`єднання...")
        self.dataFolderPathLabel.setAlignment(QtCore.Qt.AlignLeft)

        self.fileNameInfo = QLabel(self)
        self.fileNameInfo.setText("Вкажіть назву нового файлу \n"
                                  "та оберіть бажаний тип первинного опису")
        self.fileNameInfo.setAlignment(QtCore.Qt.AlignCenter)

        self.fileNameBox = QLineEdit(self)
        self.fileNameBox.setPlaceholderText("Назва Файлу...")
        self.fileNameBox.setAlignment(QtCore.Qt.AlignLeft)

        self.dsPBar = QProgressBar(self)
        self.dsPBar.setAlignment(QtCore.Qt.AlignCenter)
        self.dsPBar.setStyleSheet("QProgressBar"
                                         "{"
                                         "background-color : white;"
                                         "border : 1px solid grey;"
                                         "}")

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showThirdWindowInfo), 0, 0, 1, 2)
        layout.addWidget(QPushButton("Обрати шлях", clicked=self.getExcelFilesFolderPath), 1, 0)
        layout.addWidget(self.dataFolderPathLabel, 1, 1)

        layout.addWidget(self.fileNameInfo, 2, 0, 1, 2)
        layout.addWidget(self.fileNameBox, 3, 0, 1, 2)

        layout.addWidget(
            QPushButton("Сформувати первинний опис (Амп і фаза окремо)", clicked=self.createDataSet), 4, 0, 1, 2)
        layout.addWidget(
            QPushButton("Сформувати первинний опис (Амп і фаза разом)", clicked=self.createCombinedDataSet), 5, 0, 1, 2)

        layout.addWidget(self.dsPBar, 6, 0, 1, 2)

        excelTab.setLayout(layout)
        return excelTab

    def neuralModelsTab(self):
        modelTab = QWidget()
        layout = QGridLayout()

        self.modelStudyLabel = QLabel()
        self.modelStudyLabel.setText("Навчання моделей")
        self.modelStudyLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.rawDataPathLabel = QLineEdit()
        self.rawDataPathLabel.setPlaceholderText("Шлях до файлу з даними для навчання...")
        self.rawDataPathLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.rawDataPathLabel.setReadOnly(True)

        self.usageDataPathLabel = QLineEdit()
        self.usageDataPathLabel.setPlaceholderText("Шлях до файлу з даними для тестування...")
        self.usageDataPathLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.usageDataPathLabel.setReadOnly(True)

        self.modelSetComboBox = QComboBox()
        self.modelSetComboBox.addItems(['Перший набір моделей',
                                        'Другий набір моделей',
                                        'Моделі з кластериацією'])

        self.modelTestUsageLabel = QLabel()
        self.modelTestUsageLabel.setText("Тествування ефективності моделей")
        self.modelTestUsageLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.modelPathLabel = QLineEdit()
        self.modelPathLabel.setPlaceholderText("Шлях до моделі для тестування...")
        self.modelPathLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.modelPathLabel.setReadOnly(True)

        self.identificationDataPathLabel = QLineEdit()
        self.identificationDataPathLabel.setPlaceholderText("Шлях до файлу з даними для розпізнавання...")
        self.identificationDataPathLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.identificationDataPathLabel.setReadOnly(True)

        self.paramComboBox = QComboBox()
        self.paramComboBox.addItems(['Середнє квадратичне'])

        self.modelPBar = QProgressBar(self)
        self.modelPBar.setAlignment(QtCore.Qt.AlignCenter)
        self.modelPBar.setStyleSheet("QProgressBar"
                                  "{"
                                  "background-color : white;"
                                  "border : 1px solid grey;"
                                  "}")

        self.useModelLabel = QLabel()
        self.useModelLabel.setText("--------")
        self.useModelLabel.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showFourthWindowInfo), 0, 0, 1, 3)

        layout.addWidget(self.modelStudyLabel, 1, 0, 1, 3)
        layout.addWidget(QPushButton("Обрати шлях", clicked=self.getDataFilePath),2, 0)
        layout.addWidget(self.rawDataPathLabel, 2, 1, 1, 2)
        layout.addWidget(QPushButton("Обрати шлях", clicked=self.getTestDataFilePath),3, 0)
        layout.addWidget(self.usageDataPathLabel, 3, 1, 1, 2)

        layout.addWidget(self.modelSetComboBox, 4, 0)
        layout.addWidget(QPushButton("Навчити набір моделей", clicked=self.studyModels), 4, 1, 1, 2)
        # layout.addWidget(QPushButton("Навчити набір моделей", clicked=self.studyModelsInitial), 4, 1)
        layout.addWidget(self.modelTestUsageLabel, 5, 0, 1, 3)

        layout.addWidget(QPushButton("Обрати шлях", clicked=self.getModelFilePath), 6, 0)
        layout.addWidget(self.modelPathLabel, 6, 1, 1, 2)
        layout.addWidget(QPushButton("Обрати набір даних", clicked=self.getIdentificationDataFilePath), 7, 0)
        layout.addWidget(self.identificationDataPathLabel, 7, 1, 1, 2)
        # layout.addWidget(QPushButton("Ідентифікація", clicked=self.useModel), 8, 0, 1, 2)

        layout.addWidget(QPushButton("Ідентифікація", clicked=self.useModel), 8, 0, 1, 3)

        layout.addWidget(self.paramComboBox, 9, 0)
        layout.addWidget(QPushButton("Обрати придатну модель", clicked=self.findUsableModel), 9, 1)
        layout.addWidget(QPushButton("Обрати найкращу модель", clicked=self.findBestModel), 9, 2)

        layout.addWidget(self.modelPBar, 10, 0, 1, 3)
        layout.addWidget(self.useModelLabel, 11, 0, 1, 3)

        modelTab.setLayout(layout)
        return modelTab

    def modelManagementTab(self):
        managementTab = QWidget()
        layout = QGridLayout()

        self.operation1label = QLabel()
        self.operation1label.setText("Робота з категоріями моделей")
        self.operation1label.setAlignment(QtCore.Qt.AlignCenter)

        self.modelsSetName = QLineEdit(self)
        self.modelsSetName.setPlaceholderText("Назва категорії моделей...")
        self.modelsSetName.setAlignment(QtCore.Qt.AlignLeft)

        self.choiceLabel = QLabel()
        self.choiceLabel.setText("Робота з моделями")
        self.choiceLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.additionLabel = QLabel()
        self.deletionLabel = QLabel()

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showFifthWindowInfo), 0, 0, 1, 4)

        layout.addWidget(self.operation1label, 1, 0, 1, 4)
        layout.addWidget(QPushButton("Створити категорію", clicked=self.createModelCategory), 2, 0)
        layout.addWidget(self.modelsSetName, 2, 1, 1, 3)

        layout.addWidget(QPushButton("Копіювати категорію", clicked=self.move), 3, 0, 1, 2)
        layout.addWidget(QPushButton("Видалити категорію", clicked=self.delete), 3, 2, 1, 2)

        layout.addWidget(self.choiceLabel, 4, 0, 1, 4)
        layout.addWidget(QPushButton("Додати модель", clicked=self.move), 5, 0, 1, 2)
        layout.addWidget(QPushButton("Видалити модель", clicked=self.delete), 5, 2, 1, 2)

        managementTab.setLayout(layout)
        return managementTab

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

    def showFifthWindowInfo(self):
        self.infoBox = QMessageBox()
        self.infoBox.setIcon(QMessageBox.Information)
        self.infoBox.setWindowTitle("Інформація")
        self.infoBox.setText("Процес менеджменту моделей.\n"
                             "1.Робота з категоріями моделей;\n"
                             "Введіть назву нової категорії, натисніть кнопку\n"
                             "та вкажіть місце для її зберігання;\n"
                             "Копіювання та виделаення категорій\n"
                             "за допомогою відповідних кнопок.\n"
                             "2.Робота з моделями;\n"
                             "Додавання та виделаення моделей\n"
                             "за допомогою відповідних кнопок.\n")
        self.infoBox.setStandardButtons(QMessageBox.Close)

        returnValue = self.infoBox.exec()

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

    # Paths set -----------------------------------------------------------------------------------------------------

    def getInitialFileFolder(self):
        file, check = QFileDialog.getOpenFileName(None, "Оберіть файл",
                                                  "", "All Files (*);;Python Files (*.py);;Text Files (*.txt)")
        # return files
        if check and file.endswith(".wav"):
            print(file)

            self.filepathLabel.setText(file)

            self.getOutherTree(file)
            self.consoleOutput.insertPlainText("Файл обрано... \n")
        else:
            self.consoleOutput.insertPlainText("Некоректний формат файлу! \n")

    def getFilesFolderPath(self):
        file = QFileDialog.getExistingDirectory(self, "Оберіть папку")

        # return files
        if file:
            print(file)

            self.folderPathLabel.setText(file + '/')

            self.getInnerTree(file)
            self.consoleOutput.insertPlainText("Шлях обрано... \n")

    def getExcelFilesFolderPath(self):
        file = QFileDialog.getExistingDirectory(self, "Оберіть папку")

        # return files
        if file:
            print(file)

            self.dataFolderPathLabel.setText(file + '/')

            self.getInnerTree(file)
            self.consoleOutput.insertPlainText("Шлях обрано... \n")

    def getDataFilePath(self):
        file, check = QFileDialog.getOpenFileName(None, "Оберіть файл",
                                                  "", "All Files (*);;Python Files (*.py);;Text Files (*.txt)")
        # return files
        if file:

            before, sep, after = file.rpartition(".")

            if after == "xlsx":
                self.rawDataPathLabel.setText(file)

                self.getOutherTree(file)
                self.consoleOutput.insertPlainText("Файл обрано... \n")
            else:
                self.rawDataPathLabel.setText("Некоректний формат файлу!")

    def getTestDataFilePath(self):
        file, check = QFileDialog.getOpenFileName(None, "Оберіть файл",
                                                  "", "All Files (*);;Python Files (*.py);;Text Files (*.txt)")
        # return files
        if file:

            before, sep, after = file.rpartition(".")

            if after == "xlsx":
                self.usageDataPathLabel.setText(file)

                self.getOutherTree(file)
                self.consoleOutput.insertPlainText("Файл з тестовими даними обрано... \n")
            else:
                self.usageDataPathLabel.setText("Некоректний формат файлу!")

    def getModelFilePath(self):
        file = QFileDialog.getExistingDirectory(self, "Оберіть папку")

        # return files
        if file:

            self.modelPathLabel.setText(file)

            self.getOutherTree(file)
            self.consoleOutput.insertPlainText("Модель обрано... \n")
        else:
            self.modelPathLabel.setText("Некоректний формат файлу!")

    def getIdentificationDataFilePath(self):
        file, check = QFileDialog.getOpenFileName(None, "Оберіть файл",
                                                  "", "All Files (*);;Python Files (*.py);;Text Files (*.txt)")
        # return files
        if file:

            before, sep, after = file.rpartition(".")

            if after == "xlsx" or after == "wav":
                self.identificationDataPathLabel.setText(file)

                self.getOutherTree(file)
                self.consoleOutput.insertPlainText("Файл обрано... \n")
            else:
                self.identificationDataPathLabel.setText("Некоректний формат файлу!")

    # Models management

    def createModelCategory(self):
        file = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження результату")

        if file:
            os.mkdir(file + '\ ' + self.modelsSetName.text())

            self.getOutherTree(file)
            self.consoleOutput.insertPlainText("Категорію Створено... \n")
        else:
            self.consoleOutput.insertPlainText("Оберіть папку... \n")

    def move(self):
        file = QFileDialog.getExistingDirectory(self, "Оберіть папку для копіювання")
        dest = QFileDialog.getExistingDirectory(self, "Оберіть місце приначення")

        if file and dest:
            self.copy(file, dest)
            # shutil.copytree(file, self.modelsSetPathLabel.text())

            self.getOutherTree(dest)
            self.consoleOutput.insertPlainText("Файл скопійовано... \n")
        else:
            self.consoleOutput.insertPlainText("Некоректний формат файлу або "
                                               "неможливо знайти вказаний шлях. \n")

    def delete(self):
        file = QFileDialog.getExistingDirectory(self, "Оберіть папку для видалення")

        if file:
            shutil.rmtree(file)

            self.getOutherTree(file)
            self.consoleOutput.insertPlainText("Видалено... \n")
        else:
            self.consoleOutput.insertPlainText("Некоректний формат файлу. \n")

    def copy(self, src, dst):
        if os.path.isdir(dst):
            dst = os.path.join(dst, os.path.basename(src))
        shutil.copytree(src, dst)

    # Add possibility to change error type

    def getParamWindow(self):
        param, check = QInputDialog.getDouble(
            self, 'Введіть додаткові дані', 'Введіть бажану межу параметра:', 85.0, 0, 100, 1)

        return param

    def findUsableModel(self):

        # Global params
        modelPath = self.modelPathLabel.text()
        identificationDataPath = self.identificationDataPathLabel.text()

        param = self.paramComboBox.currentIndex()
        paramValue = float(self.getParamWindow())

        if identificationDataPath and identificationDataPath != "Шлях до файлу з даними для розпізнавання..." and paramValue:

            xls = pd.ExcelFile(identificationDataPath)
            l = len(xls.sheet_names)

            self.modelPBar.setMaximum(1)
            # Check for multiple or single excel pages
            if l > 1:
                self.consoleOutput.insertPlainText("Іде обробка... \n")

                # Test Features
                ampData_features, ampData_labels, ampData_fileNames, phaseData_features, phaseData_labels, \
                phaseData_fileNames, headers = TensorDataProcessing.getMultipleSheetDataSet(identificationDataPath)

                if param == 0:
                    testResultAmp = Models.getBestUsable(modelPath, ampData_features, ampData_labels, paramValue)

                    # testResultPh = Models.useModels(modelPath, phaseData_features, phaseData_labels)
                elif param == 1:
                    pass

                # Ui
                self.useModelLabel.setText(testResultAmp)
                self.modelPBar.setValue(1)
                self.consoleOutput.insertPlainText(testResultAmp)
                self.consoleOutput.insertPlainText("Готово... \n")

                # self.testModelLabel.setText(testResultPh)
            else:
                self.consoleOutput.insertPlainText("Іде обробка... \n")

                # Test Features
                combData_features, combData_labels, \
                combData_fileNames, headers = TensorDataProcessing.getSingleSheetDataSet(identificationDataPath)

                if param == 0:
                    testResult = Models.getBestUsable(modelPath, combData_features, combData_labels, paramValue)
                elif param == 1:
                    pass

                # Ui
                self.useModelLabel.setText(testResult)
                self.modelPBar.setValue(1)
                self.consoleOutput.insertPlainText(testResult)
                self.consoleOutput.insertPlainText("Готово... \n")

            self.identificationDataPathLabel.setText("Шлях до файлу з даними для розпізнавання...")
        else:
            self.consoleOutput.insertPlainText("Введіть необхідні дані... \n")

    def findBestModel(self):

        # Global params
        modelPath = self.modelPathLabel.text()
        identificationDataPath = self.identificationDataPathLabel.text()

        param = self.paramComboBox.currentIndex()

        if identificationDataPath and identificationDataPath != "Шлях до файлу з даними для розпізнавання...":

            xls = pd.ExcelFile(identificationDataPath)
            l = len(xls.sheet_names)

            self.modelPBar.setMaximum(1)
            # Check for multiple or single excel pages
            if l > 1:
                self.consoleOutput.insertPlainText("Іде обробка... \n")

                # Test Features
                ampData_features, ampData_labels, ampData_fileNames, phaseData_features, phaseData_labels, \
                phaseData_fileNames, headers = TensorDataProcessing.getMultipleSheetDataSet(identificationDataPath)

                if param == 0:
                    testResultAmp = Models.getBestOptimal(modelPath, ampData_features, ampData_labels)

                    # testResultPh = Models.useModels(modelPath, phaseData_features, phaseData_labels)
                elif param == 1:
                    pass

                # Ui
                self.useModelLabel.setText(testResultAmp)
                self.modelPBar.setValue(1)
                self.consoleOutput.insertPlainText(testResultAmp)
                self.consoleOutput.insertPlainText("Готово... \n")

                # self.testModelLabel.setText(testResultPh)
            else:
                self.consoleOutput.insertPlainText("Іде обробка... \n")

                # Test Features
                combData_features, combData_labels, \
                combData_fileNames, headers = TensorDataProcessing.getSingleSheetDataSet(identificationDataPath)

                if param == 0:
                    testResult = Models.getBestOptimal(modelPath, combData_features, combData_labels)
                elif param == 1:
                    pass

                # Ui
                self.useModelLabel.setText(testResult)
                self.modelPBar.setValue(1)
                self.consoleOutput.insertPlainText(testResult)
                self.consoleOutput.insertPlainText("Готово... \n")

            self.identificationDataPathLabel.setText("Шлях до файлу з даними для розпізнавання...")
        else:
            self.consoleOutput.insertPlainText("Введіть необхідні дані... \n")

    # Additional data -----------------------------------------------------------------------------------------------

    def enterChunkLength(self):

        textboxValue = self.chunkLengthBox.text()

        if int(textboxValue) > 0:
            self.chunkLabel.setText(textboxValue)
            self.consoleOutput.insertPlainText("Довжину вказано... \n")
        else:
            self.chunkLabel.setText("Довжина відрізку не може бути меншою за 0мс")

    def enterRange(self):

        textboxStartValue = self.rangeStartTextbox.text()
        textboxEndValue = self.rangeEndTextbox.text()

        if int(textboxStartValue) > 0 and int(textboxEndValue) > 0:
            if int(textboxStartValue) > int(textboxEndValue):
                self.consoleOutput.insertPlainText("Початок діапазону не може бути більшим за кінець! \n")
                self.rangeLabel.setText("Error")
            else:
                self.rangeLabel.setText(textboxStartValue + " / " + textboxEndValue)
                self.consoleOutput.insertPlainText("Діапазон вказано... \n")
        else:
            self.consoleOutput.insertPlainText("Початок і кінець діапазону не можуть бути меншими за 0! \n")
            self.rangeLabel.setText("Error")

    def enterPointer(self):
        textboxValue = self.allyTextbox.text()

        if int(textboxValue) > -1 and int(textboxValue) < 101:
            self.allyLabel.setText(textboxValue)
            self.consoleOutput.insertPlainText("Вказівник додано... \n")
        else:
            self.allyLabel.setText("Вказівник не повинен виходити за встановлені межі")

    # Final processing -----------------------------------------------------------------------------------------------

    def processFile(self):
        filePath = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження результату")

        if filePath:

            before, sep, after = self.filepathLabel.text().rpartition(".")

            if after == "wav":
                self.consoleOutput.insertPlainText("Іде обробка... \n")

                Processing.splitAudioFile(self.filepathLabel.text(),
                                          int(self.chunkLabel.text()),
                                          filePath, self.initialPBar, self.consoleOutput)

                self.consoleOutput.insertPlainText("Файл поділено на частини. \n")
                self.filepathLabel.setPlaceholderText("Шлях до аудіофайлу...")
                self.chunkLabel.setText("--------")

                self.getOutherTree(filePath)
            else:
                self.consoleOutput.insertPlainText("Некоректний формат файлу. \n")

    def processAudioFiles(self):
        self.consoleOutput.insertPlainText("Іде обробка... \n")

        filePath = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження результату")

        if filePath:

            filePath = filePath + "/"

            if self.folderPathLabel.text() != "Шлях до папки з звуковими файлами..." and \
                    self.allyLabel.text() != "--------":

                filesList = [f for f in listdir(self.folderPathLabel.text()) if
                             isfile(join(self.folderPathLabel.text(), f))]
                filesList = natsort.natsorted(filesList)

                rangeStart, rangeEnd = self.rangeLabel.text().split(" / ")

                Processing.processFiles(self.folderPathLabel.text(), filesList, int(self.allyLabel.text()),
                                        int(rangeStart), int(rangeEnd), filePath,
                                        self.numericalPBar, self.consoleOutput)

                self.consoleOutput.insertPlainText("Файли оброблено! \n")

                self.folderPathLabel.setText("Шлях до папки з звуковими файлами...")
                self.rangeLabel.setText("--------")
                self.allyLabel.setText("--------")

                self.getOutherTree(filePath)
            else:
                self.consoleOutput.insertPlainText("Некоректний формат файлів для обробки! \n")
        else:
            self.consoleOutput.insertPlainText("Некоректний формат файлів для обробки! \n")

    def createDataSet(self):
        self.consoleOutput.insertPlainText("Іде обробка... \n")

        filePath = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження результату")

        if self.dataFolderPathLabel.text() != "Шлях до файлів для об`єднання...":

            filePath = filePath + "/"

            TensorDataProcessing.createDataSet(self.dataFolderPathLabel.text(), self.fileNameBox.text(),
                                               filePath, self.dsPBar)

            self.consoleOutput.insertPlainText("Набір даних створено! \n")

            self.dataFolderPathLabel.setText("Шлях до файлів для об`єднання...")
        else:
            self.consoleOutput.insertPlainText("Некоректний формат файлів для обробки! \n")

    def createCombinedDataSet(self):
        self.consoleOutput.insertPlainText("Іде обробка... \n")

        filePath = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження результату")

        if self.dataFolderPathLabel.text() != "Шлях до файлів для об`єднання...":

            filePath = filePath + "/"

            TensorDataProcessing.createCombinedDataSet(self.dataFolderPathLabel.text(), self.fileNameBox.text(),
                                                       filePath, self.dsPBar)

            self.consoleOutput.insertPlainText("Набір даних створено! \n")

            self.dataFolderPathLabel.setText("Шлях до файлів для об`єднання...")
        else:
            self.consoleOutput.insertPlainText("Некоректний формат файлів для обробки! \n")

    # Model Study ----------------------------------------------------------------------------------------------------

    def studyModels(self):

        saveFilePath = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження результату")
        filePath = self.rawDataPathLabel.text()
        testFilePath = self.usageDataPathLabel.text()
        modelPBar = self.modelPBar
        consoleOutput = self.consoleOutput

        modelsSet = self.modelSetComboBox.currentIndex()

        if filePath == " " and filePath == "Шлях до файлу з даними для навчання..." and testFilePath == "Шлях до файлу з даними для тестування...":
            self.consoleOutput.insertPlainText("Оберіть необхідні для роботи дані \n")
        else:
            # Get number of Excel sheets
            xls = pd.ExcelFile(filePath)
            l = len(xls.sheet_names)

            # Check for multiple or single excel pages
            if l > 1:
                # Study Features
                ampData_features, ampData_labels, ampData_fileNames, phaseData_features, phaseData_labels, \
                phaseData_fileNames, headers = TensorDataProcessing.getMultipleSheetDataSet(filePath)

                # Test Features
                ampData_testFeatures, ampData_testLabels, ampData_fileNames, phaseData_testFeatures, \
                phaseData_testLabels, phaseData_fileNames, test_headers = TensorDataProcessing.getMultipleSheetDataSet(
                    testFilePath)

                # Pass params for models
                Models.passParams(ampData_features, ampData_labels,
                                  ampData_testFeatures,
                                  ampData_testLabels, ampData_fileNames,
                                  saveFilePath,
                                  headers, modelPBar, consoleOutput)
            else:
                # Study Features
                combData_features, combData_labels, combData_fileNames, headers = TensorDataProcessing.getSingleSheetDataSet(
                    filePath)

                # Test Features
                combData_testFeatures, combData_testLabels, combData_fileNames, test_headers = TensorDataProcessing.getSingleSheetDataSet(
                    testFilePath)

                # Pass params for models
                Models.passParams(combData_features, combData_labels,
                                  combData_testFeatures,
                                  combData_testLabels, combData_fileNames,
                                  saveFilePath,
                                  headers, modelPBar, consoleOutput)

            thread_pool = QtCore.QThreadPool.globalInstance()

            # Models sets
            if modelsSet == 0:
                self.consoleOutput.insertPlainText("Перший набір моделей. Іде обробка... \n")
                for task in (Models.studyFirstTypeModels, Models.studySecondTypeModels):
                    worker = Models.Worker(task)
                    # Progress output to console
                    Models.infoSignal.progress.connect(self.print_progress)
                    Models.callbackSignal.callback.connect(self.print_progress)
                    # Models study results
                    worker.signals.result.connect(self.get_models_result)
                    # Final models sorting
                    worker.signals.finished.connect(self.thread_complete)

                    thread_pool.start(worker)
            elif modelsSet == 1:
                self.consoleOutput.insertPlainText("Другий набір моделей. Іде обробка... \n")
                for task in (Models.studyThirdTypeModels, Models.studyFourthTypeModels):
                    worker = Models.Worker(task)
                    # Progress output to console
                    Models.infoSignal.progress.connect(self.print_progress)
                    Models.callbackSignal.callback.connect(self.print_progress)
                    # Models study results
                    worker.signals.result.connect(self.get_models_result)
                    # Final models sorting
                    worker.signals.finished.connect(self.thread_complete)

                    thread_pool.start(worker)
            elif modelsSet == 2:
                self.consoleOutput.insertPlainText("Моделі із застосуванням кластериації. Іде обробка... \n")
                for task in (Models.studyThirdTypeModels, Models.studyFourthTypeModels):
                    worker = Models.Worker(task)
                    # Progress output to console
                    Models.infoSignal.progress.connect(self.print_progress)
                    Models.callbackSignal.callback.connect(self.print_progress)
                    # Models study results
                    worker.signals.result.connect(self.get_models_result)
                    # Final models sorting
                    worker.signals.finished.connect(self.thread_complete)

                    thread_pool.start(worker)


    # Study support methods -----------------------------------------------------------------------------

    def get_models_result(self, countSet, resultSet):
        res = []
        for i in range(len(countSet)):
            # print("%sGuess accuracy: %s; " % (count[i], resultSet[i]))
            self.consoleOutput.insertPlainText(f"{countSet[i]} Guess accuracy: {resultSet[i]}; ")

    def print_progress(self, string):
        if string in self.consoleOutput.toPlainText():
            QApplication.processEvents()
        else:
            self.consoleOutput.insertPlainText(f"{string} \n")
            QApplication.processEvents()

    def thread_complete(self, savePath):
        print("THREAD COMPLETE!")
        self.sort_models(savePath)
        self.modelPBar.setValue(0)

    def sort_models(self, saveFilePath):
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

        self.consoleOutput.insertPlainText("\n Моделі навчано... \n")

        self.rawDataPathLabel.setText("Шлях до файлу з даними для навчання...")
        self.usageDataPathLabel.setText("Шлях до файлу з даними для тестування...")

    # Models usage

    def useModel(self):

        # Global params
        modelPath = self.modelPathLabel.text()
        identificationDataPath = self.identificationDataPathLabel.text()

        if identificationDataPath and identificationDataPath != "Шлях до файлу з даними для розпізнавання...":

            xls = pd.ExcelFile(identificationDataPath)
            l = len(xls.sheet_names)

            self.modelPBar.setMaximum(1)
            # Check for multiple or single excel pages
            if l > 1:
                self.consoleOutput.insertPlainText("Іде обробка... \n")

                # Test Features
                ampData_features, ampData_labels, ampData_fileNames, phaseData_features, phaseData_labels, \
                phaseData_fileNames, headers = TensorDataProcessing.getMultipleSheetDataSet(identificationDataPath)

                testResultAmp = Models.useOneModel(modelPath, ampData_features, ampData_labels)

                # testResultPh = Models.useModels(modelPath, phaseData_features, phaseData_labels)

                # Ui
                self.useModelLabel.setText(testResultAmp)
                self.modelPBar.setValue(1)
                self.consoleOutput.insertPlainText("Готово... \n")

                # self.testModelLabel.setText(testResultPh)
            else:
                self.consoleOutput.insertPlainText("Іде обробка... \n")

                # Test Features
                combData_features, combData_labels, \
                combData_fileNames, headers = TensorDataProcessing.getSingleSheetDataSet(identificationDataPath)

                testResult = Models.useOneModel(modelPath, combData_features, combData_labels)

                # Ui
                self.useModelLabel.setText(testResult)
                self.modelPBar.setValue(1)
                self.consoleOutput.insertPlainText("Готово... \n")

            self.identificationDataPathLabel.setText("Шлях до файлу з даними для розпізнавання...")
        else:
            self.consoleOutput.insertPlainText("Введіть необхідні дані... \n")

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



