import unittest

from App.ModelsStudy import Models
from App.DataProcession import TensorDataProcessing
import numpy

class UnitTest(unittest.TestCase):

    def workingCapacityTest(self):

        self.assertEqual(True, True)

    def testDataSetOutput(self):

        # Check format of the arrays
        dataFilePath = "D:/PythonProjects/MagRobota/testingData/numData/preamble1.xlsx"
        numpyArray = numpy.ndarray(shape=(2,2), dtype=float, order='F')

        ampData_features, ampData_labels, ampData_fileNames, \
        phaseData_features, phaseData_labels, phaseData_fileNames = TensorDataProcessing.getMultipleSheetDataSet(
            dataFilePath)

        self.assertEqual(type(ampData_features), type(numpyArray))
        self.assertEqual(type(ampData_labels), type(numpyArray))
        self.assertEqual(type(phaseData_features), type(numpyArray))
        self.assertEqual(type(phaseData_labels), type(numpyArray))
        self.assertEqual(type(ampData_fileNames), type(numpyArray))
        self.assertEqual(type(phaseData_fileNames), type(numpyArray))

    def testDataSetOutputForSingleSheetDS(self):

        # Check format of the arrays for second func
        dataFilePath = "D:/PythonProjects/MagRobota/testingData/dataSet/Komb.xlsx"
        numpyArray = numpy.ndarray(shape=(2,2), dtype=float, order='F')

        combData_features, combData_labels, combData_fileNames = TensorDataProcessing.getSingleSheetDataSet(dataFilePath)

        self.assertEqual(type(combData_features), type(numpyArray))
        self.assertEqual(type(combData_labels), type(numpyArray))
        self.assertEqual(type(combData_fileNames), type(numpyArray))

    def testModelLoading(self):

        # Model loading check
        modelPath = "D:/PythonProjects/MagRobota/testingData/Models/0.976190"

        res = Models.loadModel(modelPath)

        self.assertTrue(res)

    def testModelOutputForAlly(self):

        # Model and data should output Свій as a result
        modelPath = "D:/PythonProjects/MagRobota/testingData/Models/0.976190"
        dataFilePath = "D:/PythonProjects/MagRobota/testingData/numData/preamble1.xlsx"

        ampData_features, ampData_labels, ampData_fileNames, \
        phaseData_features, phaseData_labels, phaseData_fileNames = TensorDataProcessing.getMultipleSheetDataSet(
            dataFilePath)

        res = Models.useModels(modelPath, phaseData_features, phaseData_labels)

        self.assertEqual(res, "Свій")

    def testModelOutputForEnemy(self):

        # Model and data should output Чужий as a result
        modelPath = "D:/PythonProjects/MagRobota/testingData/Models/0.976190"
        dataFilePath = "D:/PythonProjects/MagRobota/testingData/numData/30sFile.xlsx"

        ampData_features, ampData_labels, ampData_fileNames, \
        phaseData_features, phaseData_labels, phaseData_fileNames = TensorDataProcessing.getMultipleSheetDataSet(
            dataFilePath)

        res = Models.useModels(modelPath, phaseData_features, phaseData_labels)

        self.assertEqual(res, "Чужий")


if __name__ == '__main__':
    unittest.main()
