"""

    Window for importing data from excel files

"""
import numpy
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, \
    QVBoxLayout, QStyle, QTabWidget, QFileSystemModel, QLabel

from App.ModelsStudy import Models
from App.DataProcession import DataClustering
from App.ModelsStudy import MGUAStudy as mgua

from sklearn.model_selection import train_test_split

import openpyxl
import numpy as np


class StudyDataWindow(QMainWindow):

    """
        Init method: window params, local data arrays
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.title = 'Робота з файлом Excel'
        self.width = 300
        self.height = 50
        # self.initMainWindow()
        self.initChoiceWindow()

        self.xTrain, self.yTrain, self.xTest, self.yTest = [], [], [], []

        self._line_executed = False

    """
        Main window: choose file, file info, table 
    """

    def initChoiceWindow(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.resize(self.width, self.height)

        self.layout = QVBoxLayout()

        self.layout.addWidget(QPushButton("Excel файл", self, clicked=self.initMainWindow))
        self.layout.addWidget(QPushButton(".Npy файли", self, clicked=self.initNpyWindow))

        widget = QWidget()
        widget.setContentsMargins(-1, -1, -1, 0)
        widget.setLayout(self.layout)

        self.setCentralWidget(widget)
        self.show()

    def initMainWindow(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.resize(self.width, self.height)

        self.layout = QVBoxLayout()

        self.infoLabel = QLabel()
        self.infoLabel.setText("Робота з даними")
        self.infoLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.dataLabel = QLabel()
        self.dataLabel.setText("Шлях до файлу: ---")
        self.dataLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.columnsLabel = QLabel()
        self.columnsLabel.setText("Кількість рядків: ---")
        self.columnsLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.rowsLabel = QLabel()
        self.rowsLabel.setText("Кількість стовпчиків: ---")
        self.rowsLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.excelTable = QTableWidget()

        self.layout.addWidget(self.infoLabel)
        self.layout.addWidget(QPushButton("Вказати шлях до файлу", self, clicked=self.getFilePath))

        self.layout.addWidget(self.dataLabel)

        infolayout = QHBoxLayout()
        infolayout.addWidget(self.rowsLabel)
        infolayout.addWidget(self.columnsLabel)

        self.layout.addLayout(infolayout)

        self.layout.addWidget(self.excelTable)

        self.sheetsCombo = QComboBox()
        self.sheetsCombo.addItem("Оберіть сторінку...")
        self.sheetsCombo.model().item(0).setEnabled(False)
        self.layout.addWidget(self.sheetsCombo)

        widget = QWidget()
        widget.setContentsMargins(-1, -1, -1, 0)
        widget.setLayout(self.layout)

        self.setCentralWidget(widget)
        self.show()

    def initNpyWindow(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.resize(self.width, self.height)

        self.layout = QVBoxLayout()

        self.infoLabel = QLabel()
        self.infoLabel.setText("Робота з даними")
        self.infoLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.npyXDataLabel = QLabel()
        self.npyXDataLabel.setText("Шлях до X файлу: ---")
        self.npyXDataLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.npyYDataLabel = QLabel()
        self.npyYDataLabel.setText("Шлях до Y файлу: ---")
        self.npyYDataLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.excelTable = QTableWidget()

        self.layout.addWidget(self.infoLabel)
        self.layout.addWidget(QPushButton("Обрати файл з ознаками (X)", self, clicked=self.getNpyXFilePath))
        self.layout.addWidget(self.npyXDataLabel)
        self.layout.addWidget(QPushButton("Обрати файл з вказівниками (Y)", self, clicked=self.getNpyYFilePath))
        self.layout.addWidget(self.npyYDataLabel)
        self.layout.addWidget(QPushButton("Завантажити дані", self, clicked=self.loadNpyData))

        widget = QWidget()
        widget.setContentsMargins(-1, -1, -1, 0)
        widget.setLayout(self.layout)

        self.setCentralWidget(widget)
        self.show()

    """
        Tabs list 
    """
    def initDataProcessionTabs(self):

        tabs = QTabWidget()
        tabs.addTab(self.dataSelectionTab(), "Вибір набору даних")
        tabs.addTab(self.dataSelectionByCoordsTab(), "Вибір набору даних за вказівниками")

        self.layout.addWidget(tabs)
        self.show()

    """
        Manual selection tab: checkboxes, choose data, save data 
    """
    def dataSelectionTab(self):
        generalTab = QWidget()
        innerlayout = QGridLayout()

        innerlayout.addWidget(QLabel("Оберіть один із параметрів"), 0, 1, 1, 2)

        self.dataSplitCheckbox = QCheckBox("Розділити дані на навчальну і тестову вибірки?", self)
        self.dataSplitCheckbox.setChecked(True)
        innerlayout.addWidget(self.dataSplitCheckbox, 1, 1, 1, 2)

        self.studyCheckbox = QCheckBox("Набір для навчання?", self)
        innerlayout.addWidget(self.studyCheckbox, 2, 1, 1, 2)

        self.testCheckbox = QCheckBox("Набір для тестування?", self)
        innerlayout.addWidget(self.testCheckbox, 3, 1, 1, 2)

        self.dataSplitCheckbox.stateChanged.connect(self.onStateChange)
        self.studyCheckbox.stateChanged.connect(self.onStateChange)
        self.testCheckbox.stateChanged.connect(self.onStateChange)

        innerlayout.addWidget(QPushButton("Обрати набір даних", self, clicked=self.getDataSet), 4, 1, 1, 2)
        innerlayout.addWidget(QPushButton("Зберегти", self, clicked=self.saveDataSet), 5, 1, 1, 2)

        generalTab.setLayout(innerlayout)

        return generalTab

    """
        Coordinates selection tab: lines for rows and columns, checkboxes, choose data, save data 
    """
    def dataSelectionByCoordsTab(self):
        generalTab = QWidget()
        innerlayout = QGridLayout()

        self.manualInput = QLabel(self)
        self.manualInput.setText("Введіть перший та остнанній рядки та стовпчики із бажаного масиву даних")
        self.manualInput.setAlignment(QtCore.Qt.AlignCenter)

        self.rowStart = QLineEdit(self)
        self.rowStart.setPlaceholderText("Перший рядок")
        self.rowStart.setAlignment(QtCore.Qt.AlignLeft)

        self.rowEnd = QLineEdit(self)
        self.rowEnd.setPlaceholderText("Останній рядок")
        self.rowEnd.setAlignment(QtCore.Qt.AlignLeft)

        self.columnStart = QLineEdit(self)
        self.columnStart.setPlaceholderText("Перший стовпчик")
        self.columnStart.setAlignment(QtCore.Qt.AlignLeft)

        self.columnEnd = QLineEdit(self)
        self.columnEnd.setPlaceholderText("Останній стовпчик")
        self.columnEnd.setAlignment(QtCore.Qt.AlignLeft)

        innerlayout.addWidget(self.manualInput, 0, 1, 1, 2)

        innerlayout.addWidget(QLabel("Рядки"), 1, 1, 1, 2)
        innerlayout.addWidget(self.rowStart, 2, 1)
        innerlayout.addWidget(self.rowEnd, 2, 2)

        innerlayout.addWidget(QLabel("Стовпчики"), 3, 1, 1, 2)
        innerlayout.addWidget(self.columnStart, 4, 1)
        innerlayout.addWidget(self.columnEnd, 4, 2)

        innerlayout.addWidget(QLabel("Оберіть один із параметрів"), 5, 1, 1, 2)

        self.secondDataSplitCheckbox = QCheckBox("Розділити дані на навчальну і тестову вибірки?", self)
        self.secondDataSplitCheckbox.setChecked(True)
        innerlayout.addWidget(self.secondDataSplitCheckbox, 6, 1, 1, 2)

        self.secondStudyCheckbox = QCheckBox("Набір для навчання?", self)
        innerlayout.addWidget(self.secondStudyCheckbox, 7, 1, 1, 2)

        self.secondTestCheckbox = QCheckBox("Набір для тестування?", self)
        innerlayout.addWidget(self.secondTestCheckbox, 8, 1, 1, 2)

        self.secondDataSplitCheckbox.stateChanged.connect(self.onStateChange)
        self.secondStudyCheckbox.stateChanged.connect(self.onStateChange)
        self.secondTestCheckbox.stateChanged.connect(self.onStateChange)

        innerlayout.addWidget(QPushButton("Обрати набір даних", self, clicked=self.getDataSetByCoords), 9, 1, 1, 2)
        innerlayout.addWidget(QPushButton("Зберегти", self, clicked=self.saveDataSet), 10, 1, 1, 2)

        generalTab.setLayout(innerlayout)

        return generalTab

    """
        Load data from excel into table: load sheet, get info data, export data into table
    """
    def loadData(self, path):
        self.workbook = openpyxl.load_workbook(path)
        sheet = self.workbook.active
        values = list(sheet.values)

        # for sheet in workbook.sheets:
        #     repeat all next data for each sheet

        self.sheetsCombo.clear()
        self.sheetsCombo.addItems(self.workbook.sheetnames)
        self.sheetsCombo.currentTextChanged.connect(self.loadSheet)

        self.excelTable.setRowCount(sheet.max_row)
        self.excelTable.setColumnCount(sheet.max_column)
        self.excelTable.setHorizontalHeaderLabels(values[0])

        self.columnsLabel.setText("Кількість рядків: %s" % str(sheet.max_row))
        self.rowsLabel.setText("Кількість стовпчиків: %s" % str(sheet.max_column))

        self.loadSheet(self.workbook.sheetnames[0])

        # row = 0
        # for valueRow in values[1:]:
        #     col = 0
        #     for v in valueRow:
        #         self.excelTable.setItem(row, col, QTableWidgetItem(str(v)))
        #         col += 1
        #     row += 1

    def loadNpyData(self):

        self.X = np.load(self.npyXDataLabel.text())
        self.y = np.load(self.npyYDataLabel.text())

        print("Loaded:")
        print(self.X.shape)
        print(self.y.shape)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=self.y
        )

        print(X_train)
        print(X_test)
        print(y_train)
        print(y_test)

        self.xTrain = numpy.array(X_train)
        self.yTrain = numpy.array(y_train)
        self.xTest = numpy.array(X_test)
        self.yTest = numpy.array(y_test)

    def loadSheet(self, sheetName):

        row = 0
        sheet = self.workbook[sheetName]
        values = list(sheet.values)

        self.excelTable.setHorizontalHeaderLabels(values[0])

        for valueRow in values[1:]:
            col = 0
            for v in valueRow:
                self.excelTable.setItem(row, col, QTableWidgetItem(str(v)))
                col += 1
            row += 1


    """
        Get selected data: load sheet, get rows and cols data, get y pointer, fill features and lable lists and export
        them to the export data method
    """
    def getDataSet(self):

        if self.excelTable.selectedIndexes():

            rows = sorted(set(index.row() for index in
                              self.excelTable.selectedIndexes()))
            col = sorted(set(index.column() for index in
                              self.excelTable.selectedIndexes()))

            labelPointer = self.getLabelsWindow(col)

            self.labels = []
            self.features = []

            if labelPointer != 0:
                for row in rows:
                    self.labels.append([float(self.excelTable.item(row, col[labelPointer - 1]).text())])
                    r = []
                    for c in range(col[0], col[-1]+1):
                        if c != labelPointer:
                            r.append(float(self.excelTable.item(row, c).text()))
                    self.features.append(r)

                self.exportFirstWindowData(numpy.array(self.labels), numpy.array(self.features))
            else:
                for row in rows:
                    r = []
                    for c in range(col[0], col[-1] + 1):
                        if c != labelPointer:
                            r.append(float(self.excelTable.item(row, c).text()))
                    self.features.append(r)

                self.exportFirstWindowData(numpy.array(self.labels), numpy.array(self.features))


        else:
            msg = QMessageBox()
            msg.setWindowTitle("Помилка")
            msg.setText("Оберіть дані!")
            msg.setIcon(QMessageBox.Critical)

            x = msg.exec_()

    """
        Get selected data by coords: check if all squares are filled, get rows and cols data, get y pointer, 
        fill features and lable lists and export them to the export data method
    """
    def getDataSetByCoords(self):

        if self.rowStart.text() != "Перший рядок" and self.rowEnd.text() != "Останній рядок" \
                and self.columnStart.text() != "Перший стовпчик" and self.columnEnd.text() != "Останній стовпчик":

            col = list(i for i in range(int(self.columnStart.text())-1, int(self.columnEnd.text())))
            row = list(i for i in range(int(self.rowStart.text())-1, int(self.rowEnd.text())))

            labelPointer = self.getLabelsWindow(col)

            self.labels = []
            self.features = []

            if labelPointer != 0:
                for row in range(int(self.rowStart.text())-1, int(self.rowEnd.text())):
                    self.labels.append([float(self.excelTable.item(row, col[labelPointer - 1]).text())])
                    r = []
                    for c in range(col[0], col[-1]+1):
                        if c != labelPointer:
                            r.append(float(self.excelTable.item(row, c).text()))
                    self.features.append(r)

                self.exportSecondWindowData(numpy.array(self.labels), numpy.array(self.features))
            else:
                for row in range(int(self.rowStart.text()) - 1, int(self.rowEnd.text())):
                    r = []
                    for c in range(col[0], col[-1] + 1):
                        if c != labelPointer:
                            r.append(float(self.excelTable.item(row, c).text()))
                    self.features.append(r)

                self.exportSecondWindowData(numpy.array(self.labels), numpy.array(self.features))


        else:
            msg = QMessageBox()
            msg.setWindowTitle("Помилка")
            msg.setText("Оберіть дані!")
            msg.setIcon(QMessageBox.Critical)

            x = msg.exec_()

    """
        Export data from first tab: check for checked checkbox, get split method, use method and save result
    """
    def exportFirstWindowData(self, labels, features):

        if self.dataSplitCheckbox.isChecked():
            splitMethod = self.getSplitMethodWindow()
            trainFeatures, trainLabels, testFeatures, testLabels = [], [], [], []
            if splitMethod == "Через один":
                for i in range(len(labels)):
                    if i % 2 == 0:
                        trainFeatures.append(features[i])
                        trainLabels.append(labels[i])
                    else:
                        testFeatures.append(features[i])
                        testLabels.append(labels[i])
            elif splitMethod == "Довільний набір (80:20)":

                trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(features, labels, test_size=0.2,
                                                                                        random_state=42)

                print(trainFeatures)
                print(trainLabels)

                # first = int(len(labels) * 0.8)
                # second = int(len(labels) * 0.2)
                #
                # if first + second != len(labels):
                #     second += 1
                #
                # for i in range(0, first):
                #     trainFeatures.append(features[i])
                #     trainLabels.append(labels[i])
                # for i in range(first, first+second):
                #     testFeatures.append(features[i])
                #     testLabels.append(labels[i])
            else:
                testSetLen = self.getSplitBordersInputWindow(len(labels))

                for i in range(0, len(labels)-testSetLen):
                    trainFeatures.append(features[i])
                    trainLabels.append(labels[i])
                for i in range(len(labels)-testSetLen, len(labels)):
                    testFeatures.append(features[i])
                    testLabels.append(labels[i])
            print(numpy.array(trainLabels))
            print(numpy.array(trainFeatures))
            print(numpy.array(testLabels))
            print(numpy.array(testFeatures))

            self.xTrain = numpy.array(trainFeatures)
            self.yTrain = numpy.array(trainLabels)
            self.xTest = numpy.array(testFeatures)
            self.yTest = numpy.array(testLabels)
        elif self.studyCheckbox.isChecked():
            self.xTrain = numpy.array(features)
            self.yTrain = numpy.array(labels)
        elif self.testCheckbox.isChecked():
            self.xTest = numpy.array(features)
            self.yTest = numpy.array(labels)

    """
        Export data from second tab: check for checked checkbox, get split method, use method and save result
    """
    def exportSecondWindowData(self, labels, features):

        if self.secondDataSplitCheckbox.isChecked():
            splitMethod = self.getSplitMethodWindow()
            trainFeatures, trainLabels, testFeatures, testLabels = [], [], [], []
            if splitMethod == "Через один":
                for i in range(len(labels)):
                    if i % 2 == 0:
                        trainFeatures.append(features[i])
                        trainLabels.append(labels[i])
                    else:
                        testFeatures.append(features[i])
                        testLabels.append(labels[i])
            elif splitMethod == "Довільний набір (80:20)":

                trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(features, labels, test_size=0.2, random_state=42)

                # first = int(len(labels) * 0.8)
                # second = int(len(labels) * 0.2)
                #
                # if first + second != len(labels):
                #     second += 1
                #
                # for i in range(0, first):
                #     trainFeatures.append(features[i])
                #     trainLabels.append(labels[i])
                # for i in range(first, first + second):
                #     testFeatures.append(features[i])
                #     testLabels.append(labels[i])
            else:
                testSetLen = self.getSplitBordersInputWindow(len(labels))

                for i in range(0, len(labels) - testSetLen):
                    trainFeatures.append(features[i])
                    trainLabels.append(labels[i])
                for i in range(len(labels) - testSetLen, len(labels)):
                    testFeatures.append(features[i])
                    testLabels.append(labels[i])
            print(numpy.array(trainLabels))
            print(numpy.array(trainFeatures))
            print(numpy.array(testLabels))
            print(numpy.array(testFeatures))

            self.xTrain = numpy.array(trainFeatures)
            self.yTrain = numpy.array(trainLabels)
            self.xTest = numpy.array(testFeatures)
            self.yTest = numpy.array(testLabels)
        elif self.secondStudyCheckbox.isChecked():
            self.xTrain = numpy.array(features)
            self.yTrain = numpy.array(labels)
        elif self.secondTestCheckbox.isChecked():
            self.xTest = numpy.array(features)
            self.yTest = numpy.array(labels)

    """
        Different input dialogues
    """
    def getLabelsWindow(self, col):
        labelColumn = 0
        labelColumn, check = QInputDialog.getInt(
            self, 'Введіть додаткові дані', 'Оберіть стовпчик з модельованими ознаками (Y):\n'
                                            ' - Лише обрані стовпчики, рахунок починається з 1; \n'
                                            ' - У разі пропуску цього етапу обрані дані будуть \n'
                                            '   вважатися одним набором ознак (X), та будуть \n'
                                            '   придатні лише для окремих процедур.', col[0], 1, col[-1], 1)

        return labelColumn

    def getSplitMethodWindow(self):
        items = ["Через один", "Довільний набір (80:20)", "Введення розмірності"]
        splitMethod, check = QInputDialog.getItem(
            self, 'Оберіть метод', 'Оберіть метод розділення вибірки:', items, 0, False)

        return splitMethod

    def getSplitBordersInputWindow(self, setLen):
        testSetLen, check = QInputDialog.getInt(
            self, 'Оберіть розмір', 'Вкажіть розмір перевіряючої вибірки:\n'
                                    'Обираються останні рядки.\n'
                                    'Навчальна вибірка буде сформована із залишків початкового масиву.\n'
                                    'Розмір початкового масиву: %d' % setLen, 0, 1, setLen, 1)

        return testSetLen

    def getSaveMethodWindow(self):
        items = ["Обидва масиви", "Навчання", "Тестування", "Кластеризація"]
        saveMethod, check = QInputDialog.getItem(
            self, 'Оберіть метод', 'Оберіть що ви бажаєте зберегти:', items, 0, False)

        return saveMethod

    """
        Get File Path: Dialog, set file path, load file, init table 
    """
    def getFilePath(self):
        file, check = QFileDialog.getOpenFileName(None, "Оберіть файл", "", "Excel Files (*.xlsx);")
        # return files
        if file:
            self.dataLabel.setText(file)
            self.loadData(file)

            if not self._line_executed:
                self.initDataProcessionTabs()
                self._line_executed = True

        else:
            self.dataLabel.setText("Оберіть файл!")

    def getNpyXFilePath(self):
        file, check = QFileDialog.getOpenFileName(None, "Оберіть файл", "", "NumPy Files (*.npy);")
        # return files
        if file:
            self.npyXDataLabel.setText(file)
        else:
            self.dataLabel.setText("Оберіть файл!")
    def getNpyYFilePath(self):
        file, check = QFileDialog.getOpenFileName(None, "Оберіть файл", "", "NumPy Files (*.npy);")
        # return files
        if file:
            self.npyYDataLabel.setText(file)
        else:
            self.dataLabel.setText("Оберіть файл!")

    """
        Save Data Set: choose what to save, check if arrays are full, send data to models class
    """
    def saveDataSet(self):

        print("OUT===============================")
        print("TRAIN===============================")
        print(self.xTrain)
        print(self.yTrain)
        print("TESTING===============================")
        print(self.xTest)
        print(self.yTest)

        saveMethod = self.getSaveMethodWindow()

        if saveMethod == "Обидва масиви":
            if self.xTrain == [] or self.yTrain == [] or self.xTest == [] or self.yTest == []:
                msg = QMessageBox()
                msg.setWindowTitle("Помилка")
                msg.setText("Один або декілька з масивів порожні!")
                msg.setIcon(QMessageBox.Critical)

                x = msg.exec_()
            else:
                Models.getDataParams(self.xTrain, self.yTrain, self.xTest, self.yTest)
                # mgua.getDataParams(self.xTrain, self.yTrain, self.xTest, self.yTest)
                self.xTrain, self.yTrain, self.xTest, self.yTest = [], [], [], []
        elif saveMethod == "Навчання":
            if self.xTrain == [] or self.yTrain == []:
                msg = QMessageBox()
                msg.setWindowTitle("Помилка")
                msg.setText("Один або декілька з масивів порожні!")
                msg.setIcon(QMessageBox.Critical)

                x = msg.exec_()
            else:
                Models.getDataStudyParams(self.xTrain, self.yTrain)
                # mgua.getDataStudyParams(self.xTrain, self.yTrain)
                self.xTrain, self.yTrain = [], []
        elif saveMethod == "Тестування":
            if self.xTest == [] or self.yTest == []:
                msg = QMessageBox()
                msg.setWindowTitle("Помилка")
                msg.setText("Один або декілька з масивів порожні!")
                msg.setIcon(QMessageBox.Critical)

                x = msg.exec_()
            else:
                Models.getDataTestParams(self.xTest, self.yTest)
                # mgua.getDataTestParams(self.xTest, self.yTest)
                self.xTest, self.yTest = [], []
        else:
            if self.xTrain == [] or self.yTrain == []:
                msg = QMessageBox()
                msg.setWindowTitle("Помилка")
                msg.setText("Один або декілька з масивів порожні!")
                msg.setIcon(QMessageBox.Critical)

                x = msg.exec_()
            else:
                DataClustering.getData(self.xTrain)
                self.xTrain = []

    """
        Changing checked checkbox on click: if one box is checked > uncheck others
    """
    @pyqtSlot(int)
    def onStateChange(self, state):
        if state == Qt.Checked:
            # First tab
            if self.sender() == self.dataSplitCheckbox:
                self.studyCheckbox.setChecked(False)
                self.testCheckbox.setChecked(False)
            elif self.sender() == self.studyCheckbox:
                self.dataSplitCheckbox.setChecked(False)
                self.testCheckbox.setChecked(False)
            elif self.sender() == self.testCheckbox:
                self.dataSplitCheckbox.setChecked(False)
                self.studyCheckbox.setChecked(False)
            # Second tab
            elif self.sender() == self.secondDataSplitCheckbox:
                self.secondStudyCheckbox.setChecked(False)
                self.secondTestCheckbox.setChecked(False)
            elif self.sender() == self.secondStudyCheckbox:
                self.secondDataSplitCheckbox.setChecked(False)
                self.secondTestCheckbox.setChecked(False)
            elif self.sender() == self.secondTestCheckbox:
                self.secondDataSplitCheckbox.setChecked(False)
                self.secondStudyCheckbox.setChecked(False)
