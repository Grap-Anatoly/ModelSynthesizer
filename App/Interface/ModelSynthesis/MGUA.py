"""

    Window for Model synthesis operations via MGUA (Синтез моделей, МГУА)

"""
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, \
    QVBoxLayout, QStyle, QTabWidget, QFileSystemModel, QLabel

# Additional Classes
from App.DataProcession import DataClustering
from App.Interface.ModelSynthesis import PlotOutput
from App.Interface.ModelSynthesis import NeuralModels, MGUA
from App.Interface import ExcelDataImport

from App.ModelsStudy import MGUAStudy

# OS interaction, files in folders
import os
import pandas as pd
import shutil

class MGUASynthesisWindow(QMainWindow):
    """
        Init method: window params, local data arrays
    """
    def __init__(self, parent=None):
        super(MGUASynthesisWindow, self).__init__(parent)

        self.title = 'МГУА'
        self.width = 500
        self.height = 600
        self.initModelSynthesisTabs()
    """
        Main window: choose model type
    """
    def initModelSynthesisTabs(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.setFixedSize(self.width, self.height)

        self.layout = QVBoxLayout()

        tabs = QTabWidget()
        tabs.addTab(self.modelsStudyTab(), "Синтез моделей")
        tabs.addTab(self.modelsUsageTab(), "Робота з моделями")

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

    def modelsStudyTab(self):
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

        self.selectionTypeLabel = QLabel()
        self.selectionTypeLabel.setText("Оберіть бажаний метод селекції")
        self.selectionTypeLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.selectionParamComboBox = QComboBox()
        self.selectionParamComboBox.addItems(['За кількістю х-ів в наборі даних',
                                              'Лінійно (4-та частина моделей)',
                                              'Ввести вручну'])

        self.modelTypeLabel = QLabel()
        self.modelTypeLabel.setText("Оберіть бажане опорне рівняння та синтезуйте набір моделей")
        self.modelTypeLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.modelSetComboBox = QComboBox()
        self.modelSetComboBox.addItems(['Лінійне: y = a0 + a1xi + a2xk',
                                        'Нелінійне: y = a0 + a1xi + a2xk + a3xi*xk + a4xi^2 + a5xk^2'])

        self.modelPBar = QProgressBar(self)
        self.modelPBar.setAlignment(QtCore.Qt.AlignCenter)
        self.modelPBar.setStyleSheet("QProgressBar"
                                     "{"
                                     "background-color : white;"
                                     "border : 1px solid grey;"
                                     "}")

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showModelsStudyWindowInfo), 0, 0, 1, 3)

        layout.addWidget(self.modelStudyLabel, 1, 0, 1, 3)
        layout.addWidget(QPushButton("Обрати дані для навчання", clicked=self.getStudyData), 2, 0, 1, 3)

        # layout.addWidget(QPushButton("Обрати шлях", clicked=self.getDataFilePath), 3, 0)
        # layout.addWidget(self.rawDataPathLabel, 3, 1, 1, 2)
        # layout.addWidget(QPushButton("Обрати шлях", clicked=self.getTestDataFilePath), 4, 0)
        # layout.addWidget(self.usageDataPathLabel, 4, 1, 1, 2)

        layout.addWidget(self.selectionTypeLabel, 5, 0, 1, 3)
        layout.addWidget(self.selectionParamComboBox, 6, 0, 1, 3)

        layout.addWidget(self.modelTypeLabel, 7, 0, 1, 3)

        layout.addWidget(self.modelSetComboBox, 8, 0)
        layout.addWidget(QPushButton("Синтезувати набір моделей", clicked=self.studyModels), 8, 1, 1, 2)
        # layout.addWidget(QPushButton("Навчити набір моделей", clicked=self.studyModelsInitial), 4, 1)

        layout.addWidget(self.modelPBar, 9, 0, 1, 3)

        modelTab.setLayout(layout)
        return modelTab

    def modelsUsageTab(self):
        modelTab = QWidget()
        layout = QGridLayout()

        self.modelStudyLabel = QLabel()
        self.modelStudyLabel.setText("Випробування моделей")
        self.modelStudyLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.modelPathLabel = QLineEdit()
        self.modelPathLabel.setPlaceholderText("Шлях до файлу з даними моделі...")
        self.modelPathLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.modelPathLabel.setReadOnly(True)

        self.modelPBar = QProgressBar(self)
        self.modelPBar.setAlignment(QtCore.Qt.AlignCenter)
        self.modelPBar.setStyleSheet("QProgressBar"
                                     "{"
                                     "background-color : white;"
                                     "border : 1px solid grey;"
                                     "}")

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showModelsUsageWindowInfo), 0, 0, 1, 3)

        layout.addWidget(self.modelStudyLabel, 1, 0, 1, 3)
        layout.addWidget(QPushButton("Обрати дані для випробування", clicked=self.getStudyData), 2, 0, 1, 3)

        # layout.addWidget(QPushButton("Обрати шлях", clicked=self.getDataFilePath), 3, 0)
        # layout.addWidget(self.rawDataPathLabel, 3, 1, 1, 2)
        # layout.addWidget(QPushButton("Обрати шлях", clicked=self.getTestDataFilePath), 4, 0)
        # layout.addWidget(self.usageDataPathLabel, 4, 1, 1, 2)

        layout.addWidget(QPushButton("Обрати модель для випробування", clicked=self.getModelPath), 3, 0, 1, 3)
        layout.addWidget(self.modelPathLabel, 4, 0, 1, 3)

        layout.addWidget(QPushButton("Випробувати модель", clicked=self.testModel), 5, 0, 1, 3)
        # layout.addWidget(QPushButton("Навчити набір моделей", clicked=self.studyModelsInitial), 4, 1)

        layout.addWidget(self.modelPBar, 6, 0, 1, 3)

        modelTab.setLayout(layout)
        return modelTab

    def showModelsStudyWindowInfo(self):
        self.infoBox = QMessageBox()
        self.infoBox.setIcon(QMessageBox.Information)
        self.infoBox.setWindowTitle("Інформація")
        self.infoBox.setText("Процес навчання моделей.\n"
                             "Навчання:\n"
                             "1.Вкажіть шлях до бажаного файлу з даними;\n"
                             "(Первинний опис, створений раніше)\n"
                             "2.Оберіть набір даних для навчання, та для перевірки;\n"
                             "3.Вкажіть місце для збереження навчаних моделей\n"
                             "і натисніть на кнопку;\n"
                             "(Для збереження моделей\n"
                             "рекомендується створити окрему папку)")

        self.infoBox.setStandardButtons(QMessageBox.Close)

        returnValue = self.infoBox.exec()

    def showModelsUsageWindowInfo(self):
        self.infoBox = QMessageBox()
        self.infoBox.setIcon(QMessageBox.Information)
        self.infoBox.setWindowTitle("Інформація")
        self.infoBox.setText("Процес випробування моделей.\n"
                             "1.Вкажіть шлях до бажаного файлу з даними;\n"
                             "(Первинний опис, створений раніше)\n"
                             "2.Оберіть набір даних для перевірки;\n"
                             "3.Вкажіть місце до файлу з навченою моделлю МГУА\n"
                             "і натисніть на кнопку;\n"
                             "(Для збереження моделей\n"
                             "рекомендується створити окрему папку)")

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


    # Paths set -----------------------------------------------------------------------------------------------------

    def getModelPath(self):
        file, check = QFileDialog.getOpenFileName(None, "Оберіть файл",
                                                  "", "Text Files (*.txt)")

        # return files
        if file:

            self.modelPathLabel.setText(file)

            self.getOutherTree(file)
            self.consoleOutput.insertPlainText("Модель обрано... \n")
        else:
            self.modelPathLabel.setText("Некоректний формат файлу!")

    # Get data ------------------------------------------------------------------------------------------------------

    def getStudyData(self):
        ExcelDataImport.StudyDataWindow(self).show()

    # Clusterisation ------------------------------------------------------------------------------------------------

    def getClustersNumber(self):
        sse, shilouetteCoefficients = DataClustering.getClustersNumber()

        PlotOutput.ClusterNumberWindow(sse, shilouetteCoefficients).show()

    def getClusters(self):
        n = self.clustersTextBox.text()
        if n != "":

            cY, cX = DataClustering.getClusters(int(n))

            PlotOutput.PlotOutputWindow(cY, cX).show()

            self.consoleOutput.insertPlainText("Кластеризовані ваги: \n")
            self.consoleOutput.insertPlainText(str(cY))

            results = []
            for i in range(len(cY)):
                results.append([cY[0], cX[0]])

            results = results.sort()
        else:
            self.consoleOutput.insertPlainText("Введіть необхідні дані... \n")
    # Study ---------------------------------------------------------------------------------------------------------

    def studyModels(self):

        saveFilePath = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження результату")
        filePath = self.rawDataPathLabel.text()
        testFilePath = self.usageDataPathLabel.text()
        modelPBar = self.modelPBar
        consoleOutput = self.consoleOutput

        selectionParam = self.selectionParamComboBox.currentIndex()
        modelsSet = self.modelSetComboBox.currentIndex()

        if selectionParam == 2:
            selectionParam = float(self.getParamWindow())

        if filePath == " " and filePath == "Шлях до файлу з даними для навчання..." and testFilePath == "Шлях до файлу з даними для тестування...":
            self.consoleOutput.insertPlainText("Оберіть необхідні для роботи дані \n")
        else:
            MGUAStudy.getAssistantValues(saveFilePath, modelPBar, consoleOutput, selectionParam)

            if modelsSet == 0:
                MGUAStudy.studyModel()

    def testModel(self):

        saveFilePath = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження результату")
        filePath = self.rawDataPathLabel.text()
        testFilePath = self.usageDataPathLabel.text()
        modelPBar = self.modelPBar
        consoleOutput = self.consoleOutput

        if filePath == " " and filePath == "Шлях до файлу з даними для навчання..." and testFilePath == "Шлях до файлу з даними для тестування...":
            self.consoleOutput.insertPlainText("Оберіть необхідні для роботи дані \n")
        else:
            """ PLACEHOLDER """
            """ CHECK MODELS HERE"""
            pass

    def getParamWindow(self):
        param, check = QInputDialog.getInt(
            self, 'Введіть додаткові дані', 'Введіть бажану кількість моделей для нового ряду селекції:',
            0, 0, 1000, 10)

        return param

