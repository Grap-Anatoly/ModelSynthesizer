"""

    Window for Audio Data Set Creation

"""
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, \
    QVBoxLayout, QStyle, QTabWidget, QFileSystemModel, QLabel

# Additional Classes
from App.DataProcession import Processing, TensorDataProcessing

# Folders name sorting
import natsort

# OS interaction, files in folders
from os import listdir
from os.path import isfile, join


class FileProcessionWindow(QMainWindow):

    def __init__(self, parent=None):
        super(FileProcessionWindow, self).__init__(parent)

        self.title = 'Формування МВД Аудіо-Файлів'
        self.width = 500
        self.height = 600
        self.initFileProcessionTabs()

    def initFileProcessionTabs(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.setFixedSize(self.width, self.height)

        self.layout = QVBoxLayout()

        tabs = QTabWidget()
        tabs.addTab(self.initialFileProcessionTab(), "Розбиття початкового файлу")
        tabs.addTab(self.numericalDataProcessionTab(), "Отримання числових даних")
        tabs.addTab(self.numericalDataNpyProcessionTab(), "Отримання числових даних (.Npy)")
        tabs.addTab(self.dataSetProcessionTab(), "Формування первинного опису")

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

        self.partInfo = QLabel(self)
        self.partInfo.setText("Введіть початок та кінець бажаного файлу, мс")
        self.partInfo.setAlignment(QtCore.Qt.AlignCenter)

        self.partStart = QLineEdit(self)
        self.partStart.setPlaceholderText("Початок")
        self.partStart.setAlignment(QtCore.Qt.AlignLeft)

        self.partEnd = QLineEdit(self)
        self.partEnd.setPlaceholderText("Кінець")
        self.partEnd.setAlignment(QtCore.Qt.AlignLeft)

        self.partLabel = QLabel()
        self.partLabel.setText("--------")
        self.partLabel.setAlignment(QtCore.Qt.AlignCenter)

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

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showFirstWindowInfo), 0, 0, 1, 3)
        layout.addWidget(QPushButton("Обрати файл", clicked=self.getInitialFileFolder), 1, 0)
        layout.addWidget(self.filepathLabel, 1, 1, 1, 2)

        layout.addWidget(self.partInfo, 2, 0, 1, 3)
        layout.addWidget(QPushButton("Ввести", clicked=self.enterPartsLength), 3, 0)
        layout.addWidget(self.partStart, 3, 1)
        layout.addWidget(self.partEnd, 3, 2)

        layout.addWidget(QPushButton("Розділити файл", clicked=self.cutFile), 4, 0, 1, 3)

        layout.addWidget(self.chunkInfo, 5, 0, 1, 3)
        layout.addWidget(QPushButton("Ввести", clicked=self.enterChunkLength), 6, 0)
        layout.addWidget(self.chunkLengthBox, 6, 1, 1, 2)

        layout.addWidget(QPushButton("Розбити файл", clicked=self.processFile), 7, 0, 1, 3)

        layout.addWidget(self.initialPBar, 8, 0, 1, 3)

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

    def numericalDataNpyProcessionTab(self):
        audioTabNpy = QWidget()
        layout = QGridLayout()

        self.folderPathLabelNpy = QLineEdit(self)
        self.folderPathLabelNpy.setPlaceholderText("Шлях до папки з звуковими файлами...")
        self.folderPathLabelNpy.setAlignment(QtCore.Qt.AlignLeft)
        self.folderPathLabelNpy.setReadOnly(True)

        self.rangeInfoNpy = QLabel(self)
        self.rangeInfoNpy.setText("Введіть діапазон виведення даних, через кому, Гц")
        self.rangeInfoNpy.setAlignment(QtCore.Qt.AlignCenter)

        self.rangeStartTextboxNpy = QLineEdit(self)
        self.rangeStartTextboxNpy.setText("16")

        self.rangeEndTextboxNpy = QLineEdit(self)
        self.rangeEndTextboxNpy.setText("350")

        self.rangeLabelNpy = QLabel()
        self.rangeLabelNpy.setText("--------")

        self.allyInfoNpy = QLabel(self)
        self.allyInfoNpy.setText("Введіть 'свій або чужий', 100/0 ")
        self.allyInfoNpy.setAlignment(QtCore.Qt.AlignCenter)

        self.allyTextboxNpy = QLineEdit(self)
        self.allyTextboxNpy.setText("100")
        self.allyTextboxNpy.setAlignment(QtCore.Qt.AlignLeft)

        self.allyLabelNpy = QLabel()
        self.allyLabelNpy.setText("--------")

        self.numericalPBarNpy = QProgressBar(self)
        self.numericalPBarNpy.setAlignment(QtCore.Qt.AlignCenter)
        self.numericalPBarNpy.setStyleSheet("QProgressBar"
                                         "{"
                                         "background-color : white;"
                                         "border : 1px solid grey;"
                                         "}")

        layout.addWidget(QPushButton("Інформація по процесу", clicked=self.showNpyWindowInfo), 0, 0, 1, 3)
        layout.addWidget(QPushButton("Обрати папку", clicked=self.getFilesFolderPathNpy), 1, 0)
        layout.addWidget(self.folderPathLabelNpy, 1, 1, 1, 2)

        layout.addWidget(self.rangeInfoNpy, 2, 0, 1, 3)
        layout.addWidget(QPushButton('Ввести', clicked=self.enterRangeNpy), 3, 0)
        layout.addWidget(self.rangeStartTextboxNpy, 3, 1)
        layout.addWidget(self.rangeEndTextboxNpy, 3, 2)

        layout.addWidget(self.allyInfoNpy, 4, 0, 1, 3)
        layout.addWidget(QPushButton('Ввести', clicked=self.enterPointerNpy), 5, 0)
        layout.addWidget(self.allyTextboxNpy, 5, 1, 1, 2)

        # layout.addWidget(QPushButton("Перетворити", clicked=self.longRunProcessFiles), 6, 0, 1, 3)
        layout.addWidget(QPushButton("Перетворити", clicked=self.processAudioFilesNpy), 6, 0, 1, 3)
        layout.addWidget(self.numericalPBarNpy, 7, 0, 1, 3)

        audioTabNpy.setLayout(layout)
        return audioTabNpy

    def dataSetProcessionTab(self):
        excelTab = QWidget()
        layout = QGridLayout()

        self.dataFolderPathLabel = QLineEdit()
        self.dataFolderPathLabel.setPlaceholderText("Шлях до файлів для об`єднання...")
        self.dataFolderPathLabel.setAlignment(QtCore.Qt.AlignLeft)

        self.fileNameInfo = QLabel(self)
        self.fileNameInfo.setText("Вкажіть назву нового файлу \n"
                                  "та оберіть бажаний вигляд збережених даинх")
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

    # Info ---------------------------------------------------------------------------------------------------------

    def showFirstWindowInfo(self):
        self.infoBox = QMessageBox()
        self.infoBox.setIcon(QMessageBox.Information)
        self.infoBox.setWindowTitle("Інформація")
        self.infoBox.setText("Процес виділення відрізку аудіофайлу.\n"
                             "1.Вкажіть шлях до аудоіфайлу.\n"
                             "2.Вкажіть бажані межі в мілісекундах.\n"
                             "3.Оберіть місце для збереження нових файлів і\n"
                             "натисніть на кнопку 'Розділити файл';\n"
                             "Процес розбиття аудіофайлу на частини.\n"
                             "1.Вкажіть аудіофайл для подальшого розбиття;\n"
                             "2.Оберіть бажаний процес:"
                             "Виділення одного відрізку бажаної довжини;\n"
                             "Розбиття файлів на декілька відрізків бажаної довжини;\n"
                             "3.Вкажіть бажану тривалість відрізку в мілісекундах;\n"
                             "або\n"
                             "Вкажіть бажану довжину нових файлів в мілісекундах;\n"
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

    def showNpyWindowInfo(self):
        self.infoBox = QMessageBox()
        self.infoBox.setIcon(QMessageBox.Information)
        self.infoBox.setWindowTitle("Інформація")
        self.infoBox.setText("Процес перетворення аудіофайлу в числові дані (Формат .Npy).\n"
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

    def getFilesFolderPathNpy(self):
        file = QFileDialog.getExistingDirectory(self, "Оберіть папку")

        # return files
        if file:
            print(file)

            self.folderPathLabelNpy.setText(file + '/')

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

    # Additional data -----------------------------------------------------------------------------------------------

    def enterPartsLength(self):

        startVal = self.partStart.text()
        endVal = self.partEnd.text()

        if int(startVal) > 0 and int(endVal) > 0:
            if int(startVal) < int(endVal):
                vals = startVal + ',' + endVal
                self.partLabel.setText(vals)
                self.consoleOutput.insertPlainText("Довжину вказано... \n")
            else:
                self.consoleOutput.insertPlainText("Часова мітка кінця має бути більшою за початок")
        else:
            self.consoleOutput.insertPlainText("Початок або кінець відрізку не може бути меншим за 0мс")

    def enterChunkLength(self):

        textboxValue = self.chunkLengthBox.text()

        if int(textboxValue) > 0:
            self.chunkLabel.setText(textboxValue)
            self.consoleOutput.insertPlainText("Довжину вказано... \n")
        else:
            self.consoleOutput.insertPlainText("Довжина відрізку не може бути меншою за 0мс")


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

    def enterRangeNpy(self):

        textboxStartValueNpy = self.rangeStartTextboxNpy.text()
        textboxEndValueNpy = self.rangeEndTextboxNpy.text()

        if int(textboxStartValueNpy) > 0 and int(textboxEndValueNpy) > 0:
            if int(textboxStartValueNpy) > int(textboxEndValueNpy):
                self.consoleOutput.insertPlainText("Початок діапазону не може бути більшим за кінець! \n")
                self.rangeLabelNpy.setText("Error")
            else:
                self.rangeLabelNpy.setText(textboxStartValueNpy + " / " + textboxEndValueNpy)
                self.consoleOutput.insertPlainText("Діапазон вказано... \n")
        else:
            self.consoleOutput.insertPlainText("Початок і кінець діапазону не можуть бути меншими за 0! \n")
            self.rangeLabelNpy.setText("Error")


    def enterPointer(self):
        textboxValue = self.allyTextbox.text()

        if int(textboxValue) > -1 and int(textboxValue) < 101:
            self.allyLabel.setText(textboxValue)
            self.consoleOutput.insertPlainText("Вказівник додано... \n")
        else:
            self.allyLabel.setText("Вказівник не повинен виходити за встановлені межі")

    def enterPointerNpy(self):
        textboxValue = self.allyTextboxNpy.text()

        if int(textboxValue) > -1 and int(textboxValue) < 101:
            self.allyLabelNpy.setText(textboxValue)
            self.consoleOutput.insertPlainText("Вказівник додано... \n")
        else:
            self.allyLabelNpy.setText("Вказівник не повинен виходити за встановлені межі")

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

    def cutFile(self):

        filePath = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження результату")

        if filePath:

            before, sep, after = self.filepathLabel.text().rpartition(".")

            if after == "wav":
                self.consoleOutput.insertPlainText("Іде обробка... \n")

                start, s, end = self.partLabel.text().rpartition(",")

                print(start)
                print(end)

                Processing.cutAudioFile(self.filepathLabel.text(),
                                        filePath, int(start), int(end),
                                        self.consoleOutput)

                self.consoleOutput.insertPlainText("Файл розділено. \n")
                self.filepathLabel.setPlaceholderText("Шлях до аудіофайлу...")
                self.partLabel.setText("--------")

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


    def processAudioFilesNpy(self):
        self.consoleOutput.insertPlainText("Іде обробка... \n")

        filePath = QFileDialog.getExistingDirectory(self, "Оберіть папку для збереження результату")

        if filePath:

            filePath = filePath + "/"

            if self.folderPathLabelNpy.text() != "Шлях до папки з звуковими файлами..." and \
                    self.allyLabelNpy.text() != "--------":

                filesList = [f for f in listdir(self.folderPathLabelNpy.text()) if
                             isfile(join(self.folderPathLabelNpy.text(), f))]
                filesList = natsort.natsorted(filesList)

                rangeStart, rangeEnd = self.rangeLabelNpy.text().split(" / ")

                Processing.processFilesNpy(self.folderPathLabelNpy.text(), filesList, int(self.allyLabelNpy.text()),
                                           int(rangeStart), int(rangeEnd), filePath,
                                           self.numericalPBar, self.consoleOutput)

                self.consoleOutput.insertPlainText("Файли оброблено! \n")

                self.folderPathLabelNpy.setText("Шлях до папки з звуковими файлами...")
                self.rangeLabelNpy.setText("--------")
                self.allyLabelNpy.setText("--------")

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

