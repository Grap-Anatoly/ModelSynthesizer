import os

import pandas as pd
import numpy

from PyQt5.QtWidgets import QApplication

# Read data

def createDataSet(filesPath, fileName, savePath, pBar):
    path = filesPath
    # filesPath = os.path.abspath('')
    files = os.listdir(path)

    print(files)
    QApplication.processEvents()

    ampData = pd.DataFrame()
    phaseData = pd.DataFrame()

    pBar.setMaximum(len(files))
    c = 1
    for file in files:
        if file.endswith('.xlsx'):
            xls = pd.ExcelFile(path + file)
            if 'Амп' in xls.sheet_names:
                # Change usecols A to B if you need to ignore fileNames
                ampData = ampData.append(
                    xls.parse('Амп', skiprows=0, usecols='A:XFD', index_col=None, na_values=['NA']),
                    ignore_index=True)
                phaseData = phaseData.append(
                    xls.parse('Фаза', skiprows=0, usecols='A:XFD', index_col=None, na_values=['NA']),
                    ignore_index=True)
        # Progress bar
        pBar.setValue(c)
        c += 1
        QApplication.processEvents()

    ampData.head()
    phaseData.head()

    with pd.ExcelWriter(f'{savePath + fileName}.xlsx') as writer:
        ampData.to_excel(writer, sheet_name='Амп', index=False)
        phaseData.to_excel(writer, sheet_name='Фаза', index=False)


def createCombinedDataSet(filesPath, fileName, savePath, pBar):
    path = filesPath
    # filesPath = os.path.abspath('')
    files = os.listdir(path)

    print(files)

    ampData = pd.DataFrame()
    phaseData = pd.DataFrame()

    pBar.setMaximum(len(files))
    c = 1
    for file in files:
        if file.endswith('.xlsx'):
            xls = pd.ExcelFile(path + file)
            if 'Амп'in xls.sheet_names:
                # Change usecols A to B if you need to ignore fileNames
                ampData = ampData.append(
                    xls.parse('Амп', skiprows=0, usecols='A:XFD', index_col=None, na_values=['NA']),
                    ignore_index=True)
                phaseData = phaseData.append(
                    xls.parse('Фаза', skiprows=0, usecols='A:XFD', index_col=None, na_values=['NA']),
                    ignore_index=True)
        # Progress bar
        pBar.setValue(c)
        c += 1
        QApplication.processEvents()

    ampData.head()
    phaseData.head()

    phaseData.pop('Свій/Чужий')

    combined = ampData.join(phaseData.set_index('Файл'), on='Файл').dropna().drop_duplicates()

    with pd.ExcelWriter(f'{savePath + fileName}.xlsx') as writer:
        combined.to_excel(writer, sheet_name='Комб', index=False)

def createNpyDataSet(filesPath, fileName, savePath, pBar):
    path = filesPath
    files = os.listdir(path)

    QApplication.processEvents()
    X_list = []
    y_list = []

    for file in sorted(os.listdir(path)):

        full_path = os.path.join(path, file)

        if file.startswith("X") and file.endswith(".npy"):
            X_list.append(numpy.load(full_path))

        elif file.startswith("y") and file.endswith(".npy"):
            y_list.append(numpy.load(full_path))

    feature_count = X_list[0].shape[1]

    for dataset in X_list:

        if dataset.shape[1] != feature_count:
            raise ValueError(
                f"Feature mismatch: "
                f"{dataset.shape[1]} != {feature_count}"
            )

    X = numpy.concatenate(X_list, axis=0)
    y = numpy.concatenate(y_list, axis=0)

    numpy.save(os.path.join(savePath, f"X_{fileName}.npy"), X)
    numpy.save(os.path.join(savePath, f"y_{fileName}.npy"), y)

    pBar.setValue(100)

    print("Combined X:", X.shape)
    print("Combined y:", y.shape)

# --------------------------------------------------
def getMultipleSheetDataSet(fileName):

    ampData = pd.DataFrame()
    phaseData = pd.DataFrame()

    xls = pd.ExcelFile(fileName)


    ampData = ampData.append(xls.parse('Амп', usecols='A:XFD', index_col=None, na_values=['NA']),
                             ignore_index=True)
    phaseData = phaseData.append(xls.parse('Фаза', usecols='A:XFD', index_col=None, na_values=['NA']),
                                 ignore_index=True)

    # Amplitude

    ampData_features = ampData.copy()

    ampData_labels = ampData_features.pop('Свій/Чужий')
    ampData_fileNames = ampData_features.pop('Файл')

    ampData_features = numpy.array(ampData_features)
    ampData_labels = numpy.array(ampData_labels)
    ampData_fileNames = numpy.array(ampData_fileNames)

    # Phase

    phaseData_features = phaseData.copy()

    phaseData_labels = phaseData_features.pop('Свій/Чужий')
    phaseData_fileNames = phaseData_features.pop('Файл')

    phaseData_features = numpy.array(phaseData_features)
    phaseData_labels = numpy.array(phaseData_labels)
    phaseData_fileNames = numpy.array(phaseData_fileNames)

    headers = ampData.columns

    xls.close()

    return ampData_features, ampData_labels, ampData_fileNames, phaseData_features,\
           phaseData_labels, phaseData_fileNames, headers


def getSingleSheetDataSet(fileName):

    combData = pd.DataFrame()

    xls = pd.ExcelFile(fileName)

    combData = combData.append(xls.parse('Комб', usecols='A:XFD', index_col=None, na_values=['NA']),
                               ignore_index=True)

    combData_features = combData.copy()

    combData_labels = combData_features.pop('Свій/Чужий')
    combData_fileNames = combData_features.pop('Файл')

    combData_features = numpy.array(combData_features)
    combData_labels = numpy.array(combData_labels)
    combData_fileNames = numpy.array(combData_fileNames)

    headers = combData.columns

    xls.close()

    return combData_features, combData_labels, combData_fileNames, headers

# --------------------------------------------------

# https://www.tensorflow.org/tutorials/load_data/csv
# https://keras.io/guides/sequential_model/
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
