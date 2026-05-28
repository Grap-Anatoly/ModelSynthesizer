"""

    Window for Model testing operations (Випробування моделей)

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


class ModelTestingWindow(QMainWindow):

    def __init__(self, parent=None):
        super(ModelTestingWindow, self).__init__(parent)

        self.title = 'Випробування моделей'
        self.width = 500
        self.height = 600
        self.initModelTestingTabs()

    def initModelTestingTabs(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.setFixedSize(self.width, self.height)

        self.layout = QVBoxLayout()

        tabs = QTabWidget()
        tabs.addTab(self.firstTestingTab(), "Тестування моделей")

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

    def firstTestingTab(self):
        modelTab = QWidget()
        layout = QGridLayout()

        self.modelTestUsageLabel = QLabel()
        self.modelTestUsageLabel.setText("Тествування ефективності моделей")
        self.modelTestUsageLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.rawDataPathLabel = QLineEdit()
        self.rawDataPathLabel.setPlaceholderText("Шлях до файлу із істинними данними...")
        self.rawDataPathLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.rawDataPathLabel.setReadOnly(True)

        self.modelPathLabel = QLineEdit()
        self.modelPathLabel.setPlaceholderText("Шлях до моделі для тестування...")
        self.modelPathLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.modelPathLabel.setReadOnly(True)

        self.identificationDataPathLabel = QLineEdit()
        self.identificationDataPathLabel.setPlaceholderText("Шлях до файлу з даними для розпізнавання...")
        self.identificationDataPathLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.identificationDataPathLabel.setReadOnly(True)

        self.infoLabel = QLabel()
        self.infoLabel.setText("Оберіть бажаний параметер для випробувань")
        self.infoLabel.setAlignment(QtCore.Qt.AlignCenter)

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

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showTestWindowInfo), 0, 0, 1, 3)

        layout.addWidget(self.modelTestUsageLabel, 1, 0, 1, 3)

        layout.addWidget(QPushButton("Обрати модель", clicked=self.getModelFilePath), 2, 0)
        layout.addWidget(self.modelPathLabel, 2, 1, 1, 2)

        layout.addWidget(QPushButton("Обрати дані для тестування", clicked=self.getStudyData), 4, 0, 1, 3)

        layout.addWidget(QPushButton("Ідентифікація", clicked=self.useModel), 5, 0, 1, 3)

        layout.addWidget(self.infoLabel, 6, 0, 1, 3)

        layout.addWidget(QPushButton("Точніть моделі", clicked=self.findModelAccuracy), 7, 0)
        layout.addWidget(QPushButton("Адекватність моделі", clicked=self.findModelAdequacy), 7, 1)
        layout.addWidget(QPushButton("Стійкість моделі", clicked=self.findModelStability), 7, 2)

        # layout.addWidget(QPushButton("Ідентифікація", clicked=self.useModel), 4, 0, 1, 3)

        layout.addWidget(self.modelPBar, 8, 0, 1, 3)
        layout.addWidget(self.useModelLabel, 9, 0, 1, 3)

        modelTab.setLayout(layout)
        return modelTab

    # Info ----------------------------------------------------------------------------------------------------------

    def showTestWindowInfo(self):
        self.infoBox = QMessageBox()
        self.infoBox.setIcon(QMessageBox.Information)
        self.infoBox.setWindowTitle("Інформація")
        self.infoBox.setText("Процес випрбування моделей.\n"
                             "1.Вкажіть шлях до папки з бажаною моделю;\n"
                             "(Необхідно обрати папку з усіма файлами моделі)\n"
                             "2.Вкажіть шлях до бажаного файлу з коректними даними;\n"
                             "3.Вкажіть шлях до бажаного файлу з тестовими даними;\n"
                             "3.Оберіть бажаний параметр для випробувань.")
        self.infoBox.setStandardButtons(QMessageBox.Close)

        returnValue = self.infoBox.exec()

    # Folder Tree Output -------------------------------------------------------------------------------------------

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

    # Models testing ------------------------------------------------------------------------------------------------

    def findModelAccuracy(self):

        # Global params
        modelPath = self.modelPathLabel.text()

        if Models.checkTestData() != False:

            self.modelPBar.setMaximum(1)

        # if identificationDataPath and identificationDataPath != "Шлях до файлу з даними для розпізнавання...":
        #
        #     xls = pd.ExcelFile(identificationDataPath)
        #     l = len(xls.sheet_names)
        #
        #     self.modelPBar.setMaximum(1)
        #     # Check for multiple or single excel pages
        #     if l > 1:
        #         self.consoleOutput.insertPlainText("Іде обробка... \n")
        #
        #         # Test Features
        #         ampData_features, ampData_labels, ampData_fileNames, phaseData_features, phaseData_labels, \
        #         phaseData_fileNames, headers = TensorDataProcessing.getMultipleSheetDataSet(identificationDataPath)
        #
        #         result = Models.getAccuracy(modelPath, ampData_features, ampData_labels)
        #
        #         # Ui
        #         self.useModelLabel.setText(result)
        #         self.modelPBar.setValue(1)
        #         self.consoleOutput.insertPlainText(f"{result} \n")
        #         self.consoleOutput.insertPlainText("Готово... \n")
        #
        #         # self.testModelLabel.setText(testResultPh)
        #     else:
        #         self.consoleOutput.insertPlainText("Іде обробка... \n")
        #
        #         # Test Features
        #         combData_features, combData_labels, \
        #         combData_fileNames, headers = TensorDataProcessing.getSingleSheetDataSet(identificationDataPath)
        #
        #         result = Models.getAccuracy(modelPath, combData_features, combData_labels)
        #
        #         # Ui
        #         self.useModelLabel.setText(result)
        #         self.modelPBar.setValue(1)
        #         self.consoleOutput.insertPlainText(f"{result} \n")
        #         self.consoleOutput.insertPlainText("Готово... \n")

            result = Models.getAccuracy(modelPath)

            # Ui
            self.useModelLabel.setText(result)
            self.modelPBar.setValue(1)
            self.consoleOutput.insertPlainText("Обрахунок точності моделі... \n")
            self.consoleOutput.insertPlainText(f"{result} \n")
            self.consoleOutput.insertPlainText("Готово... \n")

            self.identificationDataPathLabel.setText("Шлях до файлу з даними для розпізнавання...")
        else:
            self.consoleOutput.insertPlainText("Введіть необхідні дані... \n")

    def findModelAdequacy(self):

        # Global params
        modelPath = self.modelPathLabel.text()
        initaialDataPath = self.rawDataPathLabel.text()
        identificationDataPath = self.identificationDataPathLabel.text()

        if identificationDataPath and identificationDataPath != "Шлях до файлу з даними для розпізнавання...":
            pass

    def findModelStability(self):

        # Global params
        modelPath = self.modelPathLabel.text()
        # identificationDataPath = self.identificationDataPathLabel.text()

        if Models.checkTestData() != False:

            self.modelPBar.setMaximum(1)

        # if identificationDataPath and identificationDataPath != "Шлях до файлу з даними для розпізнавання...":
        #     xls = pd.ExcelFile(identificationDataPath)
        #     l = len(xls.sheet_names)
        #
        #     self.modelPBar.setMaximum(1)
        #     # Check for multiple or single excel pages
        #     if l > 1:
        #         self.consoleOutput.insertPlainText("Іде обробка... \n")
        #
        #         # Test Features
        #         ampData_features, ampData_labels, ampData_fileNames, phaseData_features, phaseData_labels, \
        #             phaseData_fileNames, headers = TensorDataProcessing.getMultipleSheetDataSet(identificationDataPath)
        #
        #         result = Models.getStability(modelPath, ampData_features, ampData_labels)
        #
        #         # Ui
        #         self.useModelLabel.setText("Готово")
        #         self.modelPBar.setValue(1)
        #
        #         for r in result:
        #             self.consoleOutput.insertPlainText(f"{r} \n")
        #
        #         self.consoleOutput.insertPlainText("Готово... \n")
        #
        #         # self.testModelLabel.setText(testResultPh)
        #     else:
        #         self.consoleOutput.insertPlainText("Іде обробка... \n")
        #
        #         # Test Features
        #         combData_features, combData_labels, \
        #             combData_fileNames, headers = TensorDataProcessing.getSingleSheetDataSet(identificationDataPath)
        #
        #         result = Models.getStability(modelPath, combData_features, combData_labels)
        #
        #         # Ui
        #         self.useModelLabel.setText("Готово")
        #         self.modelPBar.setValue(1)
        #
        #         for r in result:
        #             self.consoleOutput.insertPlainText(f"{r} \n")
        #
        #         self.consoleOutput.insertPlainText("Готово... \n")

            result = Models.getStability(modelPath)

            # Ui
            self.useModelLabel.setText("Перевірка стійкості моделі...\n")
            self.modelPBar.setValue(1)

            for r in result:
                self.consoleOutput.insertPlainText(f"{r} \n")

            self.consoleOutput.insertPlainText("Готово... \n")

            self.identificationDataPathLabel.setText("Шлях до файлу з даними для розпізнавання...")
        else:
            self.consoleOutput.insertPlainText("Введіть необхідні дані... \n")

    # Best model ----------------------------------------------------------------------------------------------------

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

        if Models.checkTestData() != False:
            self.modelPBar.setMaximum(1)

        # if identificationDataPath and identificationDataPath != "Шлях до файлу з даними для розпізнавання..." and paramValue:
        #
        #     xls = pd.ExcelFile(identificationDataPath)
        #     l = len(xls.sheet_names)
        #
        #     self.modelPBar.setMaximum(1)
        #     # Check for multiple or single excel pages
        #     if l > 1:
        #         self.consoleOutput.insertPlainText("Іде обробка... \n")
        #
        #         # Test Features
        #         ampData_features, ampData_labels, ampData_fileNames, phaseData_features, phaseData_labels, \
        #         phaseData_fileNames, headers = TensorDataProcessing.getMultipleSheetDataSet(identificationDataPath)
        #
        #         if param == 0:
        #             testResultAmp = Models.getBestUsable(modelPath, ampData_features, ampData_labels, paramValue)
        #
        #             # testResultPh = Models.useModels(modelPath, phaseData_features, phaseData_labels)
        #         elif param == 1:
        #             pass
        #
        #         # Ui
        #         self.useModelLabel.setText(testResultAmp)
        #         self.modelPBar.setValue(1)
        #         self.consoleOutput.insertPlainText(testResultAmp)
        #         self.consoleOutput.insertPlainText("Готово... \n")
        #
        #         # self.testModelLabel.setText(testResultPh)
        #     else:
        #         self.consoleOutput.insertPlainText("Іде обробка... \n")
        #
        #         # Test Features
        #         combData_features, combData_labels, \
        #         combData_fileNames, headers = TensorDataProcessing.getSingleSheetDataSet(identificationDataPath)
        #
        #         if param == 0:
        #             testResult = Models.getBestUsable(modelPath, combData_features, combData_labels, paramValue)
        #         elif param == 1:
        #             pass
        #
        #         # Ui
        #         self.useModelLabel.setText(testResult)
        #         self.modelPBar.setValue(1)
        #         self.consoleOutput.insertPlainText(testResult)
        #         self.consoleOutput.insertPlainText("Готово... \n")

            if param == 0:
                testResult = Models.getBestUsable(param, modelPath, paramValue)
            elif param == 1:
                pass

            # Ui
            self.useModelLabel.setText(testResult)
            self.modelPBar.setValue(1)
            self.consoleOutput.insertPlainText(f"{testResult} \n")

            self.consoleOutput.insertPlainText("Готово... \n")

            self.identificationDataPathLabel.setText("Шлях до файлу з даними для розпізнавання...")
        else:
            self.consoleOutput.insertPlainText("Введіть необхідні дані... \n")

    def findBestModel(self):

        # Global params
        modelPath = self.modelPathLabel.text()
        identificationDataPath = self.identificationDataPathLabel.text()

        param = self.paramComboBox.currentIndex()

        if Models.checkTestData() != False:
            self.modelPBar.setMaximum(1)

        # if identificationDataPath and identificationDataPath != "Шлях до файлу з даними для розпізнавання...":
        #
        #     xls = pd.ExcelFile(identificationDataPath)
        #     l = len(xls.sheet_names)
        #
        #     self.modelPBar.setMaximum(1)
        #     # Check for multiple or single excel pages
        #     if l > 1:
        #         self.consoleOutput.insertPlainText("Іде обробка... \n")
        #
        #         # Test Features
        #         ampData_features, ampData_labels, ampData_fileNames, phaseData_features, phaseData_labels, \
        #         phaseData_fileNames, headers = TensorDataProcessing.getMultipleSheetDataSet(identificationDataPath)
        #
        #         if param == 0:
        #             testResultAmp = Models.getBestOptimal(modelPath, ampData_features, ampData_labels)
        #
        #             # testResultPh = Models.useModels(modelPath, phaseData_features, phaseData_labels)
        #         elif param == 1:
        #             pass
        #
        #         # Ui
        #         self.useModelLabel.setText(testResultAmp)
        #         self.modelPBar.setValue(1)
        #         self.consoleOutput.insertPlainText(testResultAmp)
        #         self.consoleOutput.insertPlainText("Готово... \n")
        #
        #         # self.testModelLabel.setText(testResultPh)
        #     else:
        #         self.consoleOutput.insertPlainText("Іде обробка... \n")
        #
        #         # Test Features
        #         combData_features, combData_labels, \
        #         combData_fileNames, headers = TensorDataProcessing.getSingleSheetDataSet(identificationDataPath)
        #
        #         if param == 0:
        #             testResult = Models.getBestOptimal(modelPath, combData_features, combData_labels)
        #         elif param == 1:
        #             pass
        #
        #         # Ui
        #         self.useModelLabel.setText(testResult)
        #         self.modelPBar.setValue(1)
        #         self.consoleOutput.insertPlainText(testResult)
        #         self.consoleOutput.insertPlainText("Готово... \n")
            if param == 0:
                testResult = Models.getBestOptimal(param, modelPath)
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
