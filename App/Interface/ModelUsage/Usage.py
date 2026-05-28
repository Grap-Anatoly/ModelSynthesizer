"""

    Window for Model usage operations (Використання моделей)

"""
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, \
    QVBoxLayout, QStyle, QTabWidget, QFileSystemModel, QLabel

# Additional Classes
from App.DataProcession import Processing, TensorDataProcessing
from App.ModelsStudy import Models

from App.Interface import ExcelDataImport

import pandas as pd

# OS interaction, files in folders
import os
import pandas as pd
import shutil

class ModelUsageWindow(QMainWindow):

    def __init__(self, parent=None):
        super(ModelUsageWindow, self).__init__(parent)

        self.title = 'Використання моделей'
        self.width = 500
        self.height = 600
        self.initModelManagementTabs()

    def initModelManagementTabs(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.setFixedSize(self.width, self.height)

        self.layout = QVBoxLayout()

        tabs = QTabWidget()
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

        self.modelUsageLabel = QLabel()
        self.modelUsageLabel.setText("Використання моделей")
        self.modelUsageLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.modelPathLabel = QLineEdit()
        self.modelPathLabel.setPlaceholderText("Шлях до моделі...")
        self.modelPathLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.modelPathLabel.setReadOnly(True)

        self.identificationDataPathLabel = QLineEdit()
        self.identificationDataPathLabel.setPlaceholderText("Шлях до файлу з даними для розпізнавання...")
        self.identificationDataPathLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.identificationDataPathLabel.setReadOnly(True)

        self.exportCheckBox = QCheckBox("Експортувати результати ідентифікації? ")
        self.exportCheckBox.setChecked(False)

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

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showFifthWindowInfo), 0, 0, 1, 4)

        layout.addWidget(self.operation1label, 1, 0, 1, 4)
        layout.addWidget(QPushButton("Створити категорію", clicked=self.createModelCategory), 2, 0)
        layout.addWidget(self.modelsSetName, 2, 1, 1, 3)

        layout.addWidget(QPushButton("Копіювати категорію", clicked=self.move), 3, 0, 1, 2)
        layout.addWidget(QPushButton("Видалити категорію", clicked=self.delete), 3, 2, 1, 2)

        layout.addWidget(self.choiceLabel, 4, 0, 1, 4)
        layout.addWidget(QPushButton("Додати модель", clicked=self.move), 5, 0, 1, 2)
        layout.addWidget(QPushButton("Видалити модель", clicked=self.delete), 5, 2, 1, 2)

        layout.addWidget(self.modelUsageLabel, 6, 0, 1, 4)

        layout.addWidget(QPushButton("Обрати шлях", clicked=self.getModelFilePath), 7, 0)
        layout.addWidget(self.modelPathLabel, 7, 1, 1, 3)

        layout.addWidget(QPushButton("Обрати набір даних", clicked=self.getStudyData), 8, 0, 1, 4)

        layout.addWidget(self.exportCheckBox, 9, 1, 1, 3)

        layout.addWidget(QPushButton("Ідентифікація", clicked=self.useModel), 10, 0, 1, 4)

        layout.addWidget(self.modelPBar, 11, 0, 1, 4)
        layout.addWidget(self.useModelLabel, 12, 0, 1, 4)
        managementTab.setLayout(layout)
        return managementTab

    # Info Text -----------------------------------------------------------------------------------------------------

    def showFifthWindowInfo(self):
        self.infoBox = QMessageBox()
        self.infoBox.setIcon(QMessageBox.Information)
        self.infoBox.setWindowTitle("Інформація")
        self.infoBox.setText("Процес менеджменту моделей.\n"
                             "1.Робота з категоріями моделей;\n"
                             "Введіть назву нової категорії, натисніть кнопку\n"
                             "та вкажіть місце для її зберігання;\n"
                             "Копіювання та видалення категорій\n"
                             "за допомогою відповідних кнопок.\n"
                             "2.Робота з моделями;\n"
                             "Додавання та видалення моделей\n"
                             "за допомогою відповідних кнопок.\n"
                             "Робота:\n"
                             "1.Вкажіть шлях до папки з бажаною моделю;\n"
                             "(Необхідно обрати папку з усіма файлами моделі)\n"
                             "2.Вкажіть шлях до бажаного файлу з даними для аналізу;\n"
                             "3.Натисніть на кнопку 'Використати модель'.")
        self.infoBox.setStandardButtons(QMessageBox.Close)

        returnValue = self.infoBox.exec()

    # Folder Tree Output ------------------------------------------------------------------------------------------

    def getOutherTree(self, path):
        # For file selection
        before, sep, after = path.rpartition("/")
        self.model.setRootPath(before + sep)
        self.tree.setRootIndex(self.model.index(before + sep))

    def getInnerTree(self, path):
        # For folder selection
        self.model.setRootPath(path + '/')
        self.tree.setRootIndex(self.model.index(path + '/'))

    # Get data ------------------------------------------------------------------------------------------------------

    def getStudyData(self):
        ExcelDataImport.StudyDataWindow(self).show()

    # Models management -------------------------------------------------------------------------------------------

    def getStudyData(self):
        ExcelDataImport.StudyDataWindow(self).show()

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

    # Paths set -----------------------------------------------------------------------------------------------------

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

# Models usage ------------------------------------------------------------------------------------------------

    def useModel(self):

        # Global params
        modelPath = self.modelPathLabel.text()

        if Models.checkTestData() != False:

            self.modelPBar.setMaximum(1)
            # if identificationDataPath and identificationDataPath != "Шлях до файлу з даними для розпізнавання...":

            # xls = pd.ExcelFile(identificationDataPath)
            # l = len(xls.sheet_names)
            #
            # self.modelPBar.setMaximum(1)
            # # Check for multiple or single excel pages
            # if l > 1:
            #     self.consoleOutput.insertPlainText("Іде обробка... \n")
            #
            #     # Test Features
            #     # ampData_features, ampData_labels, ampData_fileNames, phaseData_features, phaseData_labels, \
            #     # phaseData_fileNames, headers = TensorDataProcessing.getMultipleSheetDataSet(identificationDataPath)
            #
            #     testResultAmp = Models.useOneModel(modelPath)
            #
            #     # testResultPh = Models.useModels(modelPath, phaseData_features, phaseData_labels)
            #
            #     # Ui
            #     self.useModelLabel.setText(testResultAmp)
            #     self.modelPBar.setValue(1)
            #     self.consoleOutput.insertPlainText("Готово... \n")
            #
            #     # self.testModelLabel.setText(testResultPh)
            # else:
            #     self.consoleOutput.insertPlainText("Іде обробка... \n")
            #
            #     # Test Features
            #     # combData_features, combData_labels, \
            #     # combData_fileNames, headers = TensorDataProcessing.getSingleSheetDataSet(identificationDataPath)
            #
            #     testResult = Models.useOneModel(modelPath)
            #
            #     # Ui
            #     self.useModelLabel.setText(testResult)
            #     self.modelPBar.setValue(1)
            #     self.consoleOutput.insertPlainText("Готово... \n")

            testResult = Models.useOneModel(modelPath)

            # Ui
            self.useModelLabel.setText(testResult)
            self.modelPBar.setValue(1)
            self.consoleOutput.insertPlainText("Готово... \n")

            self.identificationDataPathLabel.setText("Шлях до файлу з даними для розпізнавання...")
        else:
            self.consoleOutput.insertPlainText("Введіть необхідні дані... \n")
