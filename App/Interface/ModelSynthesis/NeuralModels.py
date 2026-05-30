"""

    Window for Model synthesis operations via neural networks (Синтез моделей, нейромережі)

"""
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, \
    QVBoxLayout, QStyle, QTabWidget, QFileSystemModel, QLabel

# Additional Classes
from App.DataProcession import DataClustering
from App.ModelsStudy import Models
from App.Interface.ModelSynthesis import PlotOutput

from App.Interface import ExcelDataImport

# OS interaction, files in folders
import os
import pandas as pd
import shutil

class AISynthesisWindow(QMainWindow):
    """
        Init method: window params, local data arrays
    """
    def __init__(self, parent=None):
        super(AISynthesisWindow, self).__init__(parent)

        self.title = 'Нейромережі'
        self.width = 500
        self.height = 600
        self.initModelSynthesisTabs()
    """
        Main window
    """
    def initModelSynthesisTabs(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.setFixedSize(self.width, self.height)

        self.layout = QVBoxLayout()

        tabs = QTabWidget()
        tabs.addTab(self.dataProcessingTab(), "Попередня обробка даних")
        tabs.addTab(self.modelsStudyTab(), "Робота з моделями")
        tabs.addTab(self.modelsSelectionTab(), "Попередній відбір моделей")

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

    def dataProcessingTab(self):
        modelTab = QWidget()
        layout = QGridLayout()

        self.modelStudyLabel = QLabel()
        self.modelStudyLabel.setText("Обробка даних")
        self.modelStudyLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.clusteringLabel = QLabel()
        self.clusteringLabel.setText("Кластеризація вхідних даних")
        self.clusteringLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.clustersNumberLabel = QLabel()
        self.clustersNumberLabel.setText("Визначення оптимальної кількості кластерів")
        self.clustersNumberLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.mainClusteringLabel = QLabel()
        self.mainClusteringLabel.setText("Основний процес кластеризації")
        self.mainClusteringLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.clustersTextBox = QLineEdit()
        self.clustersTextBox.setPlaceholderText("Введіть бажану кількість кластерів")
        self.clustersTextBox.setAlignment(QtCore.Qt.AlignCenter)


        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showDataProcessingWindowInfo), 0, 0, 1, 3)

        layout.addWidget(self.clusteringLabel, 1, 0, 1, 3)
        layout.addWidget(QPushButton("Обрати дані для кластеризації", clicked=self.getStudyData), 2, 0, 1, 3)

        layout.addWidget(self.clustersNumberLabel, 3, 0, 1, 3)
        layout.addWidget(QPushButton("Визначення кількості кластерів", clicked=self.getClustersNumber), 4, 0, 1, 3)

        layout.addWidget(self.mainClusteringLabel, 5, 0, 1, 3)
        layout.addWidget(self.clustersTextBox, 6, 0, 1, 3)
        layout.addWidget(QPushButton("Клаcтеризація", clicked=self.getClusters), 7, 0, 1, 3)

        modelTab.setLayout(layout)
        return modelTab

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

        self.modelTypeLabel = QLabel()
        self.modelTypeLabel.setText("Оберіть бажаний алгоритм синтезу")
        self.modelTypeLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.modelSetComboBox = QComboBox()
        self.modelSetComboBox.addItems(['Перший алгоритм синтезу',
                                        'Другий алгоритм синтезу',
                                        'Алгоритм з кластериацією'])

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

        layout.addWidget(self.modelTypeLabel, 5, 0, 1, 3)

        layout.addWidget(self.modelSetComboBox, 6, 0)
        layout.addWidget(QPushButton("Синтезувати набір моделей", clicked=self.studyModels), 6, 1, 1, 2)
        # layout.addWidget(QPushButton("Навчити набір моделей", clicked=self.studyModelsInitial), 4, 1)

        layout.addWidget(self.modelPBar, 7, 0, 1, 3)

        modelTab.setLayout(layout)
        return modelTab

    def modelsSelectionTab(self):
        modelTab = QWidget()
        layout = QGridLayout()

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

        self.modelSelectionCriteriaLabel = QLabel()
        self.modelSelectionCriteriaLabel.setText("Селекція моделей")
        self.modelSelectionCriteriaLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.paramComboBox = QComboBox()
        self.paramComboBox.addItems(['Критерій регулярності',
                                     'Критерій мінімума зсуву'])

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

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showModelsStudyWindowInfo), 0, 0, 1, 3)
        layout.addWidget(self.modelTestUsageLabel, 1, 0, 1, 3)

        layout.addWidget(QPushButton("Обрати шлях", clicked=self.getModelFilePath), 2, 0)
        layout.addWidget(self.modelPathLabel, 2, 1, 1, 2)
        layout.addWidget(QPushButton("Обрати дані для тестування", clicked=self.getStudyData), 3, 0, 1, 3)
        # layout.addWidget(QPushButton("Обрати набір даних", clicked=self.getIdentificationDataFilePath), 3, 0)
        # layout.addWidget(self.identificationDataPathLabel, 3, 1, 1, 2)

        layout.addWidget(QPushButton("Ідентифікація", clicked=self.useModel), 4, 0, 1, 3)

        layout.addWidget(self.modelSelectionCriteriaLabel, 5, 0, 1, 3)

        layout.addWidget(self.paramComboBox, 6, 0)
        layout.addWidget(QPushButton("Обрати придатну модель", clicked=self.findUsableModel), 6, 1)
        layout.addWidget(QPushButton("Обрати оптимальну модель", clicked=self.findBestModel), 6, 2)

        layout.addWidget(self.modelPBar, 7, 0, 1, 3)
        layout.addWidget(self.useModelLabel, 8, 0, 1, 3)

        modelTab.setLayout(layout)
        return modelTab

    # Info ----------------------------------------------------------------------------------------------------------

    def showDataProcessingWindowInfo(self):
        self.infoBox = QMessageBox()
        self.infoBox.setIcon(QMessageBox.Information)
        self.infoBox.setWindowTitle("Інформація")
        self.infoBox.setText("Процес кластеризації вхідних даних.\n"
                             "1.Вкажіть шлях до бажаного файлу з даними;\n"
                             "(Первинний опис, створений раніше)\n"
                             "2.Визначте оптимальну кількість кластерів та введіть її у відповідне поле;\n"
                             "3.Натисніть на відповідну кнопку та отримайте результат кластеризації.")

        self.infoBox.setStandardButtons(QMessageBox.Close)

        returnValue = self.infoBox.exec()

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

    def showModelsSelectionWindowInfo(self):
        self.infoBox = QMessageBox()
        self.infoBox.setIcon(QMessageBox.Information)
        self.infoBox.setWindowTitle("Інформація")
        self.infoBox.setText("Процес селекції моделей.\n"
                             "Робота:\n"
                             "1.Вкажіть шлях до папки з бажаною моделю;\n"
                             "(Необхідно обрати папку з усіма файлами моделі)\n"
                             "2.Вкажіть шлях до бажаного файлу з даними для аналізу;\n"
                             "3.Натисніть на кнопку 'Використати модель'.\n"
                             "Визначення найкращої моделі.\n"
                             "1.Вкажіть шлях до папки з моделями;\n"
                             "2.Оберіть бажаний критерій;\n"
                             "3.Натисніть на кнопку з бажаним типом пошуку\n"
                             "та введіть додаткові данні, якщо необхідно.")

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


    # Best model ----------------------------------------------------------------------------------------------------

    def getParamWindow(self):
        param, check = QInputDialog.getDouble(
            self, 'Введіть додаткові дані', 'Введіть бажану межу параметра:', 85.0, 0, 100, 1)

        return param

    def findUsableModel(self):

        # Global params
        modelPath = self.modelPathLabel.text()

        param = self.paramComboBox.currentIndex()
        paramValue = float(self.getParamWindow())

        if Models.checkTestData() is not False and paramValue:

            self.modelPBar.setMaximum(1)
            self.modelPBar.setValue(0)

            testResult = Models.getBestUsable(param, modelPath, paramValue)

            # Ui
            self.useModelLabel.setText(testResult)
            self.modelPBar.setValue(1)
            self.consoleOutput.insertPlainText("Готово... \n")
            self.consoleOutput.insertPlainText("%s \n" % testResult)

            self.identificationDataPathLabel.setText("Шлях до файлу з даними для розпізнавання...")
        else:
            self.consoleOutput.insertPlainText("Введіть необхідні дані... \n")

    def findBestModel(self):

        # Global params
        modelPath = self.modelPathLabel.text()

        param = self.paramComboBox.currentIndex()

        if Models.checkTestData() is not False:

            self.modelPBar.setMaximum(1)
            self.modelPBar.setValue(0)

            testResult = Models.getBestOptimal(param, modelPath)

            # Ui
            self.useModelLabel.setText(testResult)
            self.modelPBar.setValue(1)
            self.consoleOutput.insertPlainText("Готово... \n")
            self.consoleOutput.insertPlainText("%s \n" % testResult)

        else:
            self.consoleOutput.insertPlainText("Введіть необхідні дані... \n")

    # Model Study ---------------------------------------------------------------------------------------------------

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
            # # Get number of Excel sheets
            # xls = pd.ExcelFile(filePath)
            # l = len(xls.sheet_names)
            #
            # # Check for multiple or single excel pages
            # if l > 1:
            #     # Study Features
            #     ampData_features, ampData_labels, ampData_fileNames, phaseData_features, phaseData_labels, \
            #     phaseData_fileNames, headers = TensorDataProcessing.getMultipleSheetDataSet(filePath)
            #
            #     # Test Features
            #     ampData_testFeatures, ampData_testLabels, ampData_fileNames, phaseData_testFeatures, \
            #     phaseData_testLabels, phaseData_fileNames, test_headers = TensorDataProcessing.getMultipleSheetDataSet(
            #         testFilePath)
            #
            #     # Pass params for models
            #     Models.passParams(ampData_features, ampData_labels,
            #                       ampData_testFeatures,
            #                       ampData_testLabels,
            #                       ampData_fileNames, saveFilePath,
            #                       headers, modelPBar, consoleOutput)
            # else:
            #     # Study Features
            #     combData_features, combData_labels, combData_fileNames, headers = TensorDataProcessing.getSingleSheetDataSet(
            #         filePath)
            #
            #     # Test Features
            #     combData_testFeatures, combData_testLabels, combData_fileNames, test_headers = TensorDataProcessing.getSingleSheetDataSet(
            #         testFilePath)
            #
            #     # Pass params for models
            #     Models.passParams(combData_features, combData_labels,
            #                       combData_testFeatures,
            #                       combData_testLabels,
            #                       combData_fileNames, saveFilePath,
            #                       headers, modelPBar, consoleOutput)

            Models.getAssistantValues(saveFilePath, modelPBar, consoleOutput)

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
                for task in (Models.studyFirstTypeClusteringModels, Models.studySecondTypeClusteringModels):
                    worker = Models.Worker(task)
                    # Progress output to console
                    Models.infoSignal.progress.connect(self.print_progress)
                    Models.callbackSignal.callback.connect(self.print_progress)
                    # Models study results
                    worker.signals.result.connect(self.get_models_result)
                    # Final models sorting
                    worker.signals.finished.connect(self.thread_complete)

                    thread_pool.start(worker)

    # Study support methods ----------------------------------------------------------------------------------------

    def get_models_result(self, countSet, resultSet):
        for i in range(len(countSet)):
            # print("%sGuess accuracy: %s; " % (count[i], resultSet[i]))
            self.consoleOutput.insertPlainText(f"{countSet[i]}")

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
        # models = os.listdir(saveFilePath)
        models = filter(os.path.isdir, os.listdir(saveFilePath))

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

        files = filter(os.path.isdir, os.listdir(saveFilePath))
        # for filename in os.listdir(saveFilePath):
        for filename in files:
            if filename not in bestModels:
                shutil.rmtree(saveFilePath + "/" + filename, ignore_errors=False, onerror=None)

        self.consoleOutput.insertPlainText("\n Моделі навчано... \n")

        self.rawDataPathLabel.setText("Шлях до файлу з даними для навчання...")
        self.usageDataPathLabel.setText("Шлях до файлу з даними для тестування...")

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