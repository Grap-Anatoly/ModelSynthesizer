import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import traceback, sys, os
import numpy

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from keras import layers
# Tensorboard
from keras.callbacks import TensorBoard

from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore

import csv
import io

from datetime import datetime

print(tf.__version__)

# https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
# https://towardsdatascience.com/a-comprehensive-introduction-to-tensorflows-sequential-api-and-model-for-deep-learning-c5e31aee49fa
# https://habr.com/ru/post/482126/
# https://www.digitalocean.com/community/tutorials/how-to-build-a-deep-learning-model-to-predict-employee-retention-using-keras-and-tensorflow

# Get global params for all models
def passParams(trainDataFeatures, trainDataLabels, testDataFeatures, testDataLabels,
               initialFileNames, resultSavePath, initialHeaders, progresBar, consoleOutputField):

    global trainFeatures, testFeatures, trainLabels, testLabels, \
        fileNames, savePath, headers, pBar, consoleOutput

    trainFeatures = trainDataFeatures
    trainLabels = trainDataLabels
    testFeatures = testDataFeatures
    testLabels = testDataLabels
    fileNames = initialFileNames
    savePath = resultSavePath
    headers = initialHeaders
    pBar = progresBar
    consoleOutput = consoleOutputField


# Signals class
class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(list, list)
    progress = QtCore.pyqtSignal(str)

    callback = QtCore.pyqtSignal(str)


# Thread procession class
class Worker(QtCore.QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            countSet, resultSet = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(countSet, resultSet)  # Return the result of the processing
        finally:
            self.signals.finished.emit(savePath)  # Done


infoSignal = WorkerSignals()
callbackSignal = WorkerSignals()


# Custom callback class
class CustomCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        print(
            "\n Finished Epoch № {}, Loss: {:.4f}, Accuracy {:.4f}".format(epoch+1, logs["loss"], logs["accuracy"]))
        callbackSignal.callback.emit(
            "Finished Epoch № {}, Loss: {:.4f}, Accuracy {:.4f}".format(epoch+1, logs["loss"], logs["accuracy"]))

        QApplication.processEvents()

# Voices tested (100,75,50)
# studyFirstTypeModels studyVoiceTypeModels
@QtCore.pyqtSlot()
def studyModels():

    modelType = "Downloaded"
    tensorboard = TensorBoard(log_dir=f'D:/PythonProjects/MagRobota/ModelsResults/{modelType}-{int(time.time())}')

    infoSignal.progress.emit(f'Model: {modelType}, training started!')

    # Start and end params for full study cycle
    model_start = headers[2]
    model_start = model_start.split("Гц")[0]
    model_end = headers[-1]
    model_end = model_end.split("Гц")[0]

    # Data preparation
    normalize = layers.Normalization()
    normalize.adapt(trainFeatures)

    resultSet = []
    countSet = []

    x_train = numpy.reshape(trainFeatures, (trainFeatures.shape[0], trainFeatures.shape[1],1))
    x_test = numpy.reshape(testFeatures, (testFeatures.shape[0], testFeatures.shape[1],1))

    # Model ---------------------------------------------------------------------------------------------------------

    # Tf Hub version !!!!!!!!!!!!!!!!!!!!!!!!!!!
    # https://tfhub.dev

    # Load the model.
    # model = hub.load('https://tfhub.dev/google/yamnet/1')

    # Input: 3 seconds of silence as mono 16 kHz waveform samples.
    # waveform = np.zeros(3 * 16000, dtype=np.float32)

    # Run the model, check the output.
    # scores, embeddings, log_mel_spectrogram = model(waveform)
    # scores.shape.assert_is_compatible_with([None, 521])
    # embeddings.shape.assert_is_compatible_with([None, 1024])
    # log_mel_spectrogram.shape.assert_is_compatible_with([None, 64])

    # Find the name of the class with the top score when mean-aggregated across frames.
    # def class_names_from_csv(class_map_csv_text):
    #     """Returns list of class names corresponding to score vector."""
    #     class_map_csv = io.StringIO(class_map_csv_text)
    #     class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    #     class_names = class_names[1:]  # Skip CSV header
    #     return class_names

    # class_map_path = model.class_map_path().numpy()
    # class_names = class_names_from_csv(tf.io.read_file(class_map_path).numpy().decode('utf-8'))
    # print(class_names[scores.numpy().mean(axis=0).argmax()])  # Should print 'Silence'.

    # --------------------------------------------------------------------------------------------------------------

    # Open file with model code
    # Read from for example (tf.keras.Sequential...)
    # Copy code to app and use for training
    f = open(f"D:/PythonProjects/MagRobota/ModelsResults/modelpath", "a")
    # pring(f.read())
    f.close()

    # Find model
    # Find tf.keras.Sequential(...
    # Get text from Sequential and to for example fit
    # sound_model.fit
    # As a result we get Layers and other model info

    # Else, we can just got Layers From the model file

    # Else, we can download full model and just insound_model.fitut our data into it
    # but
    # Then we must relocate inner methods outside and use them from other places

    # >>>>>>>>>>>>>>>>>>

    first_layer, second_layer = 50, 50
    cycles = 5
    pBar.setMaximum(cycles)
    for i in range(cycles):
        # Layers
        sound_model = tf.keras.Sequential([
            layers.LSTM(units=first_layer, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            layers.Dropout(0.2),
            layers.LSTM(units=second_layer, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(units=50),
            layers.Dropout(0.2),
            layers.Dense(101),
        ])

        # Compile
        sound_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            optimizer=tf.optimizers.Adam(),
                            metrics=['accuracy'])
        sound_model.summary()

        # Model Study
        EPOCHS = 15

        sound_model.fit(x_train, trainLabels, epochs=EPOCHS, batch_size=2, callbacks=[CustomCallback(), tensorboard])
        QApplication.processEvents()

        # Model prediction (testing) --------------------------------------------------------------------------------
        y_pred = numpy.argmax(sound_model.predict(x_test), axis=1)
        QApplication.processEvents()
        print(y_pred)

        y_true = testLabels

        rounded = []
        for v in y_pred:
            if v > 60:
                rounded.append(100)
            else:
                rounded.append(0)

        test_acc = sum(rounded == y_true) / len(y_true)

        infoSignal.progress.emit(f'Model: {modelType}, Cycle:{i+1}, Test set accuracy: {test_acc:.0%}')
        QApplication.processEvents()
        print(f'Test set accuracy: {test_acc:.0%}')

        count = countGuesses(fileNames, y_true, y_pred)

        # All models saving

        path = "%s/%s, %f" % (savePath, modelType, test_acc)
        saveModel(sound_model, path)

        with open(path + '\params.txt', 'w') as f:
            f.write(model_start + "\n")
            f.write(model_end)

        # ------------------------------

        resultSet.append(test_acc)
        countSet.append(count)

        # Model params change
        first_layer += 10
        second_layer += 5

        # Ui
        pBar.setValue(i+1)
        QApplication.processEvents()

    return countSet, resultSet
    # 5 models

# Model manipulation ------------------------------------------------------------------------------------------------

# Count correct guesses
def countGuesses(fileNames, y_true, y_pred):
    if len(y_true) == len(fileNames):

        fileNamesDict = {}
        namesCount = []
        names = []

        for n, l in zip(fileNames, y_true):
            fileNamesDict[n] = l

        for k, v in fileNamesDict.items():
            before, sep, after = k.rpartition("_")

            names.append(before)

        counter = 0

        for n in names:
            if y_pred[counter] >= 55 and y_true[counter] == 100:
                namesCount.append(n)
                counter = counter + 1
            else:
                counter = counter + 1

        print(namesCount)

    values, counts = numpy.unique(namesCount, return_counts=True)

    res = ""

    for i in range(len(values)):
        res += "File name: %s, %d guesses; " % (values[i], counts[i])

    if res == "":
        res = "Zero guesses "

    return res

# Save and load models
def saveModel(model, savePath):
    model.save(savePath)

def loadModel(loadPath):
    new_model = tf.keras.models.load_model(loadPath, compile=False)

    return new_model

# Get params for models
def getParams(loadPath):

    model = loadModel(loadPath)

    for layer in model.layers:
        res = layer.get_output_at(0).get_shape().as_list()
        break

    return res

# Models Usage ------------------------------------------------------------------------------------------------------

# Use one model for testing
def useOneModel(loadPath, testFeatures, testLabels):
    testModel = loadModel(loadPath)

    y_pred = numpy.argmax(testModel.predict(testFeatures), axis=1)
    y_true = testLabels

    QApplication.processEvents()

    rounded = []
    for v in y_pred:
        if v > 55:
            rounded.append(100)
        else:
            rounded.append(0)

    test_acc = rounded.count(100) / len(y_true)

    print(y_pred)
    print(y_true)
    print(round(test_acc*100)) # Percentage

    if test_acc * 100 > 70.0:
        res = ("Результат: Cвій, Точність: %d, файли: %s" % (test_acc * 100, y_pred))
    else:
        res = ("Результат: Чужий, Точність: %d, файли: %s" % (test_acc * 100, y_pred))

    return res

# Use multiple models
def useModels(loadPath, testFeatures, testLabels):

    testModel = loadModel(loadPath)

    # if numpy.count_nonzero(testLabels == 0) == len(testLabels) or numpy.count_nonzero(testLabels == 100) == len(testLabels):
    y_pred = numpy.argmax(testModel.predict(testFeatures), axis=1)
    y_true = testLabels

    # test_acc = sum(y_pred == y_true) / len(y_true)

    rounded = []
    for v in y_pred:
        if v > 55:
            rounded.append(100)
        else:
            rounded.append(0)

    # test_acc = sum(rounded == y_true) / len(y_true)
    test_acc = rounded.count(100) / len(y_true)

    print(y_pred)
    print(y_true)
    print(round(test_acc*100)) # Percentage

    res = {}
    if test_acc * 100 > 70.0:
        res[(loadPath.rsplit("\\", 1)[1])] = ("Результат: Cвій, Точність: %d, файли: %s" % (test_acc * 100, y_pred))
    else:
        res[(loadPath.rsplit("\\", 1)[1])] = ("Результат: Чужий, Точність: %d, файли: %s" % (test_acc * 100, y_pred))

    return res

# Save model results to file
def exportModelsResult(savePath, modelName, result):

    now = datetime.now()
    date = now.strftime("%d_%m_%Y_%H_%M_%S")

    before, sep, after = modelName.rpartition("/")

    # f = open(f"D:/PythonProjects/MagRobota/ModelsResults/{after + dt_string}.txt", "a")
    f = open(f"{savePath}/{after + date}.txt", "a")
    f.write(result)
    f.close()

# Best models -------------------------------------------------------------------------------------------------------

# Squared Error
def getBestUsable(criteria, loadPath, testFeatures, testLabels, maxError):

    if criteria == 0:
        # Regularity Criteria
        mse = tf.keras.losses.MeanSquaredError()

        models = os.listdir(loadPath)
        res = []

        for m in models:
            testModel = loadModel(loadPath + "/" + m)

            y_pred = numpy.argmax(testModel.predict(testFeatures), axis=1)
            y_true = testLabels

            error = mse(y_true, y_pred).numpy()
            res.append(error)

            if error <= maxError:
                return m

        return f"Жодна модель не задовільняє умови: {res}"
    elif criteria == 1:
        # Minimum Offset Criteria
        models = os.listdir(loadPath)
        extractedModels = []
        res = []

        for m in models:
            extractedModels.append(loadPath + "/" + m)

        for i in range(len(extractedModels)):
            if i < len(extractedModels) - 1:
                firstModel = loadModel(extractedModels[i])
                secondModel = loadModel(extractedModels[i+1])

                firstModelYPred = numpy.argmax(firstModel.predict(testFeatures), axis=1)
                secondModelYPred = numpy.argmax(secondModel.predict(testFeatures), axis=1)

                acc = len(testLabels)
                for r in range(len(firstModelYPred)):
                    if firstModelYPred[r] != secondModelYPred[r]:
                        acc -= 1

                error = acc / len(testLabels)
                res.append(acc / len(testLabels))

                if error <= maxError:
                    before, sep, after = extractedModels[i].rpartition("/")
                    return after
            else:
                firstModel = loadModel(extractedModels[i])
                secondModel = loadModel(extractedModels[i - 1])

                firstModelYPred = numpy.argmax(firstModel.predict(testFeatures), axis=1)
                secondModelYPred = numpy.argmax(secondModel.predict(testFeatures), axis=1)

                acc = len(testLabels)
                for r in range(len(firstModelYPred)):
                    if firstModelYPred[r] != secondModelYPred[r]:
                        acc -= 1

                error = acc / len(testLabels)
                res.append(acc / len(testLabels))

                if error <= maxError:
                    before, sep, after = extractedModels[i].rpartition("/")
                    return after

            return f"Жодна модель не задовільняє умови: {res}"
    elif criteria == 2:
        # Convergence Criteria
        pass

def getBestOptimal(criteria, loadPath, testFeatures, testLabels):

    if criteria == 0:
        # Regularity Criteria
        mse = tf.keras.losses.MeanSquaredError()

        models = os.listdir(loadPath)
        res = {}

        for m in models:
            testModel = loadModel(loadPath + "/" + m)

            y_pred = numpy.argmax(testModel.predict(testFeatures), axis=1)
            y_true = testLabels

            error = mse(y_true, y_pred).numpy()
            res[m] = error

        minErr = max(res, key=res.get)

        return f"Найкраща модель: {minErr}"
    elif criteria == 1:
        # Minimum Offset Criteria
        models = os.listdir(loadPath)
        extractedModels = []
        res = {}

        for m in models:
            extractedModels.append(loadPath + "/" + m)

        for i in range(len(extractedModels)):
            if i < len(extractedModels) - 1:
                firstModel = loadModel(extractedModels[i])
                secondModel = loadModel(extractedModels[i + 1])

                firstModelYPred = numpy.argmax(firstModel.predict(testFeatures), axis=1)
                secondModelYPred = numpy.argmax(secondModel.predict(testFeatures), axis=1)

                acc = len(testLabels)
                for r in range(len(firstModelYPred)):
                    if firstModelYPred[r] != secondModelYPred[r]:
                        acc -= 1

                error = acc / len(testLabels)
                before, sep, after = extractedModels[i].rpartition("/")
                res[after] = error

            else:
                firstModel = loadModel(extractedModels[i])
                secondModel = loadModel(extractedModels[i - 1])

                firstModelYPred = numpy.argmax(firstModel.predict(testFeatures), axis=1)
                secondModelYPred = numpy.argmax(secondModel.predict(testFeatures), axis=1)

                acc = len(testLabels)
                for r in range(len(firstModelYPred)):
                    if firstModelYPred[r] != secondModelYPred[r]:
                        acc -= 1

                error = acc / len(testLabels)
                before, sep, after = extractedModels[i].rpartition("/")
                res[after] = error

        minErr = max(res, key=res.get)

        return f"Найкраща модель: {minErr}"
