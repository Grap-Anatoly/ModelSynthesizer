# Plot Output ---------------------------------------------------------------------------------------------------
import sys

import numpy
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QMainWindow, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import random

import pyqtgraph as pg

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import numpy as np

from App.Interface.ModelSynthesis import NeuralModels

# Plot style
# https://www.pythonguis.com/tutorials/plotting-pyqtgraph/

class PlotOutputWindow(QDialog):

    def __init__(self, y, x, parent=None):
        super(PlotOutputWindow, self).__init__(parent)

        self.cY = y
        self.cX = x

        self.title = 'Результат кластеризації'
        self.setWindowTitle(self.title)

        self.plot = pg.plot()
        n = len(self.cY)
        self.scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))

        clusters = [list(self.cY), self.cX]
        clusters = np.array(clusters)

        spots = [{'pos': clusters[:, i], 'data': 1}
                 for i in range(n)] + [{'pos': [0, 0], 'data': 1}]

        self.scatter.addPoints(spots)
        self.plot.addItem(self.scatter)

        layout = QVBoxLayout()
        layout.addWidget(self.plot)

        self.setLayout(layout)

        # self.UiComponents(self.cY, self.cX)

        self.exec_()

    # method for components
    # def UiComponents(self, y, x):
    #     # creating a widget object
    #     self.setWindowTitle(self.title)
    #     widget = QWidget()
    #
    #     # creating a plot window
    #     plot = pg.plot()
    #
    #     # number of points
    #     # n = 300
    #     n = len(y)
    #
    #     # creating a scatter plot item
    #     # of size = 10
    #     # using brush to enlarge the of white color with transparency is 50%
    #     scatter = pg.ScatterPlotItem(
    #         size=10, brush=pg.mkBrush(255, 255, 255, 120))
    #
    #     # getting random position
    #     pos = np.random.normal(size=(2, n), scale=1e-5)
    #     print("YYYYYYYYYYYYYYYYYYYYYYYYY")
    #     print(y)
    #     print(x)
    #     print("POSPOSPOSPOS")
    #     print(pos)
    #
    #     clusters = [list(y), x]
    #     clusters = np.array(clusters)
    #     # clusters = numpy.append(clusters, x)
    #
    #     print(clusters)
    #
    #     # creating spots using the random position
    #     # spots = [{'pos': pos[:, i], 'data': 1}
    #     #          for i in range(n)] + [{'pos': [0, 0], 'data': 1}]
    #     spots = [{'pos': clusters[:, i], 'data': 1}
    #              for i in range(n)] + [{'pos': [0, 0], 'data': 1}]
    #
    #     # adding points to the scatter plot
    #     scatter.addPoints(spots)
    #
    #     # add item to plot window
    #     # adding scatter plot item to the plot window
    #     plot.addItem(scatter)
    #
    #     # Creating a grid layout
    #     layout = QGridLayout()
    #
    #     # # minimum width value of the label
    #     # label.setMinimumWidth(130)
    #
    #     # setting this layout to the widget
    #     widget.setLayout(layout)
    #
    #     # # adding label in the layout
    #     # layout.addWidget(label, 1, 0)
    #
    #     # plot window goes on right side, spanning 3 rows
    #     layout.addWidget(plot, 0, 1, 3, 1)
    #
    #     # setting this widget as central widget of the main window
    #     self.setCentralWidget(widget)
    #     self.show()


class ClusterNumberWindow(QDialog):

    def __init__(self, sse, shilouetteCoefficients, parent=None):
        super(ClusterNumberWindow, self).__init__(parent)

        self.sseData = sse
        self.shilouetteCoefficientsData = shilouetteCoefficients

        self.title = 'Визначення оптимальної кількості кластерів'
        self.setWindowTitle(self.title)

        self.pshilouetteCoefficientsPlot = pg.PlotWidget()
        # self.setCentralWidget(self.plot_graph)
        clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.pshilouetteCoefficientsPlot.plot(clusters, self.shilouetteCoefficientsData)

        self.ssePlot = pg.PlotWidget()
        # self.setCentralWidget(self.plot_graph)
        clusters = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.ssePlot.plot(clusters, self.sseData)

        # creating a Vertical Box layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Shilouette Coefficient"))
        layout.addWidget(self.pshilouetteCoefficientsPlot)
        layout.addWidget(QLabel("SSE"))
        layout.addWidget(self.ssePlot)

        self.setLayout(layout)
        self.exec_()

