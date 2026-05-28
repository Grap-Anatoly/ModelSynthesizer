import sys
import traceback
import logging
import multiprocessing
multiprocessing.freeze_support()

#Interface
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QStyle

#Interface Classes
from App.Interface import Accumulation, UsageInterface, Synthesizer


class Main(QMainWindow):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)

        self.left = 500
        self.top = 500
        self.title = 'БМЗ'
        self.width = 200
        self.height = 100
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.layout = QVBoxLayout()

        aButton = QPushButton("Накопичення моделей", self, clicked=self.getAccumulationWindow)
        # aButton = QPushButton("Синтезатор моделей", self, clicked=self.getAccumulationWindow)
        aButton.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))

        uButton = QPushButton("Використання моделей", self, clicked=self.getUsageWindow)
        uButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        self.layout.addWidget(aButton)
        self.layout.addWidget(uButton)

        widget = QWidget()
        widget.setContentsMargins(-1, -1, -1, 0)
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        self.show()

    def getAccumulationWindow(self):
        # Synthesizer.Synthesizer(self).show()
        Accumulation.Accumulation(self).show()

    def getUsageWindow(self):
        UsageInterface.Usage(self).show()


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
    main = Main()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':

    main()