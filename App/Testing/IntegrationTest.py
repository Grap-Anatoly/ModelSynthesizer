import os

import unittest

import numpy
from App.DataProcession import TensorDataProcessing
from App.ModelsStudy import Models


class IntegrationTest(unittest.TestCase):

    def testFullProcess(self):

        # Full process of model study and usage with const data
        studyFilePath = "D:/PythonProjects/MagRobota/testingData/dataSet/Set1.xlsx"

        ampData_features, ampData_labels, ampData_fileNames, \
        phaseData_features, phaseData_labels, phaseData_fileNames = TensorDataProcessing.getMultipleSheetDataSet(studyFilePath)

        phaseData_testFeatures = [i * 1.1 for i in phaseData_features.copy()]

        phaseData_testFeatures = numpy.array(phaseData_testFeatures)
        phaseData_testLabels = phaseData_labels

        saveFilePath = "D:/PythonProjects/MagRobota/testingData/Models1/"

        count, resultSet = Models.studyFirstTypeModels(phaseData_features, phaseData_labels,
                                                       phaseData_testFeatures,
                                                       phaseData_testLabels, phaseData_fileNames, saveFilePath)

        for i in range(len(count)):
            print("%sGuess accuracy: %s; " % (count[i], resultSet[i]))

        self.assertTrue(resultSet)

        count, resultSet = Models.studySecondTypeModels(phaseData_features, phaseData_labels,
                                                        phaseData_testFeatures,
                                                        phaseData_testLabels, phaseData_fileNames, saveFilePath)

        for i in range(len(count)):
            print("%sGuess accuracy: %s; " % (count[i], resultSet[i]))

        self.assertTrue(resultSet)

        modelPath = saveFilePath
        dataFilePath = "D:/PythonProjects/MagRobota/testingData/numData/preamble1.xlsx"

        ampData_features, ampData_labels, ampData_fileNames, \
        phaseData_features, phaseData_labels, phaseData_fileNames = TensorDataProcessing.getMultipleSheetDataSet(
            dataFilePath)

        models = [name for name in os.listdir(modelPath) if os.path.isdir(os.path.join(modelPath, name))]
        print(models)
        model = modelPath + "/" + models[-2]
        res = Models.useModels(model, phaseData_features, phaseData_labels)

        self.assertEqual(res, "Свій")

if __name__ == '__main__':
    unittest.main()
