import sys
import traceback
import logging
import multiprocessing
multiprocessing.freeze_support()

from App.Interface.MVD import MVDCreation
from App.Interface.ModelSynthesis import Synthesis
from App.Interface.ModelTesting import Testing
from App.Interface.ModelUsage import Usage

from pathlib import Path

# Interface
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, \
    QVBoxLayout, QStyle, QTabWidget, QFileSystemModel, QLabel

class Synthesizer(QMainWindow):

    def __init__(self, parent=None):
        super(Synthesizer, self).__init__(parent)

        self.title = 'Синтезатор Моделей'
        self.width = 300
        self.height = 200
        self.initSynthesizer()

    def initSynthesizer(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.setFixedSize(self.width, self.height)

        self.layout = QVBoxLayout()

        self.layout.addWidget(QPushButton("Формування МВД", self, clicked=self.getFileProcessionWindow))
        self.layout.addWidget(QPushButton("Синтез моделей", self, clicked=self.getModelSynthesisWindow))
        self.layout.addWidget(QPushButton("Випробування моделей", self, clicked=self.getModelTestingWindow))
        self.layout.addWidget(QPushButton("Використання моделей", self, clicked=self.getModelUsageWindow))

        widget = QWidget()
        widget.setContentsMargins(-1, -1, -1, 0)
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        self.show()

    def getFileProcessionWindow(self):
        MVDCreation.MVDCreationWindow(self).show()

    def getModelSynthesisWindow(self):
        Synthesis.ModelSynthesisWindow(self).show()

    def getModelTestingWindow(self):
        Testing.ModelTestingWindow(self).show()

    def getModelUsageWindow(self):
        Usage.ModelUsageWindow(self).show()

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

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(Path('InterfaceStyles/MainSheet.qss').read_text())

    main = Synthesizer()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':

    main()



