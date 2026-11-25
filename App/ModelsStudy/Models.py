import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import traceback, sys, os
import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import pygad

from pygad import kerasga
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM
from keras import callbacks
# Tensorboard
# python -m tensorboard.main --logdir=ModelsResults/
from keras.callbacks import TensorBoard

from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore

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

def getAssistantValues(resultSavePath, progresBar, consoleOutputField):

    global savePath, pBar, consoleOutput

    savePath = resultSavePath
    pBar = progresBar
    consoleOutput = consoleOutputField

def getDataParams(trainDataFeatures, trainDataLabels, testDataFeatures, testDataLabels):

    global trainFeatures, testFeatures, trainLabels, testLabels

    trainFeatures = trainDataFeatures
    trainLabels = trainDataLabels
    testFeatures = testDataFeatures
    testLabels = testDataLabels

    print("MODELS==============================")
    print("TRAIN===============================")
    print(trainFeatures)
    print(trainLabels)
    print("TESTING=============================")
    print(testFeatures)
    print(testLabels)

def getDataStudyParams(trainDataFeatures, trainDataLabels):

    global trainFeatures, trainLabels

    trainFeatures = trainDataFeatures
    trainLabels = trainDataLabels

    print("MODELS==============================")
    print("TRAIN===============================")
    print(trainFeatures)
    print(trainLabels)

def getDataTestParams(testDataFeatures, testDataLabels):

    global testFeatures, testLabels

    testFeatures = testDataFeatures
    testLabels = testDataLabels

    print("MODELS==============================")
    print("TESTING=============================")
    print(testFeatures)
    print(testLabels)

def getDataParams(trainDataFeatures, trainDataLabels, testDataFeatures, testDataLabels):

    global trainFeatures, testFeatures, trainLabels, testLabels

    trainFeatures = trainDataFeatures
    trainLabels = trainDataLabels
    testFeatures = testDataFeatures
    testLabels = testDataLabels
def getAndPrepareNpyBirdsData(trainFeatures, evaluateFeatures, testFeatures, trainLables, evaluateLabels, testLables, testIds):
    global X_train, X_val, X_test, y_train, y_val, y_test, test_file_ids

    X_train = trainFeatures
    X_val = evaluateFeatures
    X_test = testFeatures
    y_train = trainLables
    y_val = evaluateLabels
    y_test = testLables
    test_file_ids = testIds

    global le
    le = LabelEncoder()

    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    from collections import Counter
    print(Counter(y_train))
    print(Counter(y_val))
    print(Counter(y_test))

    # Global Normalization
    n_features = X_train.shape[-1]

    scalers = []

    for i in range(n_features):
        scaler = StandardScaler()

        scaler.fit(
            X_train[:, :, i].reshape(-1, 1)
        )

        X_train[:, :, i] = scaler.transform(
            X_train[:, :, i].reshape(-1, 1)
        ).reshape(X_train.shape[0], X_train.shape[1])

        X_val[:, :, i] = scaler.transform(
            X_val[:, :, i].reshape(-1, 1)
        ).reshape(X_val.shape[0], X_val.shape[1])

        X_test[:, :, i] = scaler.transform(
            X_test[:, :, i].reshape(-1, 1)
        ).reshape(X_test.shape[0], X_test.shape[1])

        scalers.append(scaler)

    classes = numpy.unique(y_train)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )

    class_weights = dict(
        zip(numpy.unique(y_train), weights)
    )


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
    # 5 models
def studyFirstTypeModels():

    if checkStudyData == False:
        consoleOutput.insertPlainText("Відсутній один, або декілька необхідних для навчання наборів даних... \n")
    else:
        modelType = " Dense Mlp"
        # tensorboard = TensorBoard(log_dir=f'TensorboardResults/{modelType}-{int(time.time())}')
        infoSignal.progress.emit(f'Model: {modelType}, training started!')

        # Data preparation
        resultSet = []
        countSet = []

        X_train, X_test, y_train, y_test = trainFeatures, testFeatures, trainLabels, testLabels
        input_dim = len(X_train[0])

        # Model params
        cycles = 1
        pBar.setMaximum(cycles)
        for i in range(cycles):

            def build_mlp_model(input_dim):
                inputs = tf.keras.Input(shape=(input_dim,))
                x = tf.keras.layers.Dense(256, activation='relu')(inputs)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dense(128, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.3)(x)
                x = tf.keras.layers.Dense(64, activation='relu')(x)
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                return tf.keras.Model(inputs, outputs)

            mlpModel = build_mlp_model(input_dim)

            mlpModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            history = mlpModel.fit(X_train, y_train, batch_size=32, epochs=20, validation_split=0.2)

            # Evaluate on test set
            testLoss, testAcc = mlpModel.evaluate(X_test, y_test)

            print(f"\nTraining accuracy: {history.history['accuracy'][-1]:.4f}")
            print(f"\nTraining loss: {history.history['loss'][-1]:.4f}")
            print(f"\nTraining val accuracy: {history.history['val_accuracy'][-1]:.4f}")

            print(f"\nTest accuracy: {testAcc:.4f}")
            print(f"\nTest loss: {testLoss:.4f}")

            # Detailed classification report, Model prediction (testing)
            yPred = mlpModel.predict(X_test)
            yPred = numpy.argmax(yPred, axis=1)  # Predicted classes
            yTrue = numpy.argmax(y_test, axis=1)  # True classes

            predPercent = 0.35
            predNumber = int(len(X_test) * predPercent)
            matchNumber = 0

            randomIndices = numpy.random.choice(
                len(X_test),
                predNumber,
                replace=False
            )

            print(randomIndices)
            for idx in randomIndices:
                print(f"Sample {idx}: Predicted={yPred[idx]}, Real={yTrue[idx]}")
                if yPred[idx] == yTrue[idx]:
                    matchNumber += 1

            testAcc = (matchNumber / predNumber) * 100

            count = f"Matches: {matchNumber} of {predNumber} is true. Guess Accuracy: {testAcc}"
            print(count)

            # All models saving

            path = "%s/%s, %f" % (savePath, modelType, testAcc)
            saveModel(mlpModel, path)

            # Params for one process identification
            # with open(path + '\params.txt', 'w') as f:
            #     f.write(model_start + "\n")
            #     f.write(model_end)

            resultSet.append(testAcc)
            countSet.append(count)

            # Ui
            pBar.setValue(i+1)
            QApplication.processEvents()

        return countSet, resultSet
    # 5 models

@QtCore.pyqtSlot()
def studySecondTypeModels():

    if checkStudyData == False:
        consoleOutput.insertPlainText("Відсутній один, або декілька необхідних для навчання наборів даних... \n")
    else:
        modelType = "Convolutional"
        # tensorboard = TensorBoard(log_dir=f'TensorboardResults/{modelType}-{int(time.time())}')
        infoSignal.progress.emit(f'Model: {modelType}, training started!')

        # Data preparation
        normalize = layers.Normalization()
        normalize.adapt(trainFeatures)

        resultSet = []
        countSet = []

        scaler = StandardScaler()
        XTrainNorm = scaler.fit_transform(trainFeatures)
        XTestNorm = scaler.fit_transform(testFeatures)

        # Reshape for CNN: (samples, height=num_frequencies, width=num_features, channels=1)
        XTrainCnn = XTrainNorm.reshape(XTrainNorm.shape[0], XTrainNorm.shape[1], 1, 1)
        XTestCnn = XTestNorm.reshape(XTestNorm.shape[0], XTestNorm.shape[1], 1, 1)

        yTrainCat = to_categorical(trainLabels, num_classes=2)  # adjust num_classes if needed
        yTestCat = to_categorical(testLabels, num_classes=2)

        X_train, X_test, y_train, y_test = XTrainCnn, XTestCnn, yTrainCat, yTestCat

        for i in range(1):

            convModel = Sequential([
                Conv2D(16, (3, 1), activation='relu', padding='same', input_shape=(XTrainCnn.shape[1], 1, 1)),
                MaxPooling2D((2, 1)),
                Dropout(0.3),

                Conv2D(32, (3, 1), activation='relu', padding='same'),
                MaxPooling2D((2, 1)),
                Dropout(0.3),

                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.4),
                Dense(2, activation='softmax')
            ])

            convModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            history = convModel.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32)

            # Evaluate on test set
            testLoss, testAcc = convModel.evaluate(X_test, y_test)

            print(f"\nTraining accuracy: {history.history['accuracy'][-1]:.4f}")
            print(f"\nTraining loss: {history.history['loss'][-1]:.4f}")
            print(f"\nTraining val accuracy: {history.history['val_accuracy'][-1]:.4f}")

            print(f"\nTest accuracy: {testAcc:.4f}")
            print(f"\nTest loss: {testLoss:.4f}")

            # Detailed classification report, Model prediction (testing)
            yPred = convModel.predict(X_test)
            yPred = numpy.argmax(yPred, axis=1)  # Predicted classes
            yTrue = numpy.argmax(y_test, axis=1)  # True classes

            predPercent = 0.35
            predNumber = int(len(X_test) * predPercent)
            matchNumber = 0

            randomIndices = numpy.random.choice(
                len(X_test),
                predNumber,
                replace=False
            )

            print(randomIndices)
            for idx in randomIndices:
                print(f"Sample {idx}: Predicted={yPred[idx]}, Real={yTrue[idx]}")
                if yPred[idx] == yTrue[idx]:
                    matchNumber += 1

            testAcc = (matchNumber / predNumber) * 100

            count = f"Matches: {matchNumber} of {predNumber} is true. Guess Accuracy: {testAcc}"
            print(count)
            # All models saving

            path = "%s/%s, %f" % (savePath, modelType, testAcc)
            saveModel(convModel, path)

            # Models params for one process identifiaction
            # with open(path + '\params.txt', 'w') as f:
            #     f.write(model_start + "\n")
            #     f.write(model_end)

            resultSet.append(testAcc)
            countSet.append(count)

            # Ui
            QApplication.processEvents()

        return countSet, resultSet
        # 5 models

@QtCore.pyqtSlot()
def studyThirdTypeModels():

    if checkStudyData == False:
        consoleOutput.insertPlainText("Відсутній один, або декілька необхідних для навчання наборів даних... \n")
    else:
        modelType = "2D LSTM"
        # tensorboard = TensorBoard(log_dir=f'TensorboardResults/{modelType}-{int(time.time())}')
        infoSignal.progress.emit(f'Model: {modelType}, training started!')

        # Data preparation
        resultSet = []
        countSet = []

        scaler = StandardScaler()
        XTrainNorm = scaler.fit_transform(trainFeatures)
        XTestNorm = scaler.fit_transform(testFeatures)

        # Reshape for CNN: (samples, height=num_frequencies, width=num_features, channels=1)
        XTrainCnn = XTrainNorm.reshape(XTrainNorm.shape[0], XTrainNorm.shape[1], 1)
        XTestCnn = XTestNorm.reshape(XTestNorm.shape[0], XTestNorm.shape[1], 1)

        yTrainCat = to_categorical(trainLabels, num_classes=2)  # adjust num_classes if needed
        yTestCat = to_categorical(testLabels, num_classes=2)

        X_train, X_test, y_train, y_test = XTrainCnn, XTestCnn, yTrainCat, yTestCat

        for i in range(1):

            lstm2DModel = Sequential([
                LSTM(64, input_shape=(XTrainCnn.shape[1], 1), return_sequences=True),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.4),
                Dense(2, activation='softmax')
            ])

            lstm2DModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            history = lstm2DModel.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=32)

            # Evaluate on test set
            testLoss, testAcc = lstm2DModel.evaluate(X_test, y_test)

            print(f"\nTraining accuracy: {history.history['accuracy'][-1]:.4f}")
            print(f"\nTraining loss: {history.history['loss'][-1]:.4f}")
            print(f"\nTraining val accuracy: {history.history['val_accuracy'][-1]:.4f}")

            print(f"\nTest accuracy: {testAcc:.4f}")
            print(f"\nTest loss: {testLoss:.4f}")

            # Detailed classification report, Model prediction (testing)
            yPred = lstm2DModel.predict(X_test)
            yPred = numpy.argmax(yPred, axis=1)  # Predicted classes
            yTrue = numpy.argmax(y_test, axis=1)  # True classes

            predPercent = 0.35
            predNumber = int(len(X_test) * predPercent)
            matchNumber = 0

            randomIndices = numpy.random.choice(
                len(X_test),
                predNumber,
                replace=False
            )

            for idx in randomIndices:
                print(f"Sample {idx}: Predicted={yPred[idx]}, Real={yTrue[idx]}")
                if yPred[idx] == yTrue[idx]:
                    matchNumber += 1

            testAcc = (matchNumber / predNumber) * 100

            count = f"Matches: {matchNumber} of {predNumber} is true. Guess Accuracy: {testAcc}"
            print(count)

            # All models saving
            path = "%s/%s, %f" % (savePath, modelType, testAcc)
            saveModel(lstm2DModel, path)

            # Models params for one process identifiaction
            # with open(path + '\params.txt', 'w') as f:
            #     f.write(model_start + "\n")
            #     f.write(model_end)

            resultSet.append(testAcc)
            countSet.append(count)

            # Ui
            QApplication.processEvents()

        return countSet, resultSet
        # 5 models

# Fourth type (5 layers)

@QtCore.pyqtSlot()
def studyFourthTypeModels():
    if checkStudyData == False:
        consoleOutput.insertPlainText("Відсутній один, або декілька необхідних для навчання наборів даних... \n")
    else:
        modelType = "2D LSTM (Smaller dimmensions)"
        # tensorboard = TensorBoard(log_dir=f'TensorboardResults/{modelType}-{int(time.time())}')
        infoSignal.progress.emit(f'Model: {modelType}, training started!')

        # Data preparation
        resultSet = []
        countSet = []

        scaler = StandardScaler()
        XTrainNorm = scaler.fit_transform(trainFeatures)
        XTestNorm = scaler.fit_transform(testFeatures)

        # Reshape for CNN: (samples, height=num_frequencies, width=num_features, channels=1)
        XTrainCnn = XTrainNorm.reshape(XTrainNorm.shape[0], XTrainNorm.shape[1], 1)
        XTestCnn = XTestNorm.reshape(XTestNorm.shape[0], XTestNorm.shape[1], 1)

        yTrainCat = to_categorical(trainLabels, num_classes=2)  # adjust num_classes if needed
        yTestCat = to_categorical(testLabels, num_classes=2)

        X_train, X_test, y_train, y_test = XTrainCnn, XTestCnn, yTrainCat, yTestCat

        for i in range(1):

            lstm2DModel = Sequential([
                LSTM(32, input_shape=(XTrainCnn.shape[1], 1), return_sequences=True),
                Dropout(0.2),
                LSTM(16),
                Dropout(0.2),
                # Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dropout(0.3),
                Dense(2, activation='softmax')
            ])

            lstm2DModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            history = lstm2DModel.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=32)

            # Evaluate on test set
            testLoss, testAcc = lstm2DModel.evaluate(X_test, y_test)

            print(f"\nTraining accuracy: {history.history['accuracy'][-1]:.4f}")
            print(f"\nTraining loss: {history.history['loss'][-1]:.4f}")
            print(f"\nTraining val accuracy: {history.history['val_accuracy'][-1]:.4f}")

            print(f"\nTest accuracy: {testAcc:.4f}")
            print(f"\nTest loss: {testLoss:.4f}")

            # Detailed classification report, Model prediction (testing)
            yPred = lstm2DModel.predict(X_test)
            yPred = numpy.argmax(yPred, axis=1)  # Predicted classes
            yTrue = numpy.argmax(y_test, axis=1)  # True classes

            predPercent = 0.35
            predNumber = int(len(X_test) * predPercent)
            matchNumber = 0

            randomIndices = numpy.random.choice(
                len(X_test),
                predNumber,
                replace=False
            )

            for idx in randomIndices:
                print(f"Sample {idx}: Predicted={yPred[idx]}, Real={yTrue[idx]}")
                if yPred[idx] == yTrue[idx]:
                    matchNumber += 1

            testAcc = (matchNumber / predNumber) * 100

            count = f"Matches: {matchNumber} of {predNumber} is true. Guess Accuracy: {testAcc}"
            print(count)

            # All models saving
            path = "%s/%s, %f" % (savePath, modelType, testAcc)
            saveModel(lstm2DModel, path)

            # Models params for one process identifiaction
            # with open(path + '\params.txt', 'w') as f:
            #     f.write(model_start + "\n")
            #     f.write(model_end)

            resultSet.append(testAcc)
            countSet.append(count)

            # Ui
            QApplication.processEvents()

        return countSet, resultSet


# # Fourth type (5 layers, sparce error)
@QtCore.pyqtSlot()
def studyFifthTypeModels():

    if checkStudyData == False:
        consoleOutput.insertPlainText("Відсутній один, або декілька необхідних для навчання наборів даних... \n")
    else:
        modelType = "LSTM_5_Layers"
        tensorboard = TensorBoard(log_dir=f'TensorboardResults/{modelType}-{int(time.time())}')
        # tensorboard = TensorBoard(log_dir=f'D:/PythonProjects/MagRobota/ModelsResults/{modelType}-{int(time.time())}')

        infoSignal.progress.emit(f'Model: {modelType}, training started!')

        # Data preparation
        normalize = layers.Normalization()
        normalize.adapt(trainFeatures)

        resultSet = []
        countSet = []

        x_train = numpy.reshape(trainFeatures, (trainFeatures.shape[0], trainFeatures.shape[1], 1))
        x_test = numpy.reshape(testFeatures, (testFeatures.shape[0], testFeatures.shape[1], 1))

        # Model params
        first_layer, second_layer, third_layer, fourth_layer = 55, 50, 45, 40

        for i in range(5):
            # Layers
            sound_model = tf.keras.Sequential([
                layers.LSTM(units=first_layer, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                layers.Dropout(0.2),
                layers.LSTM(units=second_layer, return_sequences=True),
                layers.Dropout(0.2),
                layers.LSTM(units=third_layer, return_sequences=True),
                layers.Dropout(0.2),
                layers.LSTM(units=fourth_layer, return_sequences=True),
                layers.Dropout(0.2),
                layers.LSTM(units=30),
                layers.Dropout(0.2),
                layers.Dense(101),
            ])

            # Model compile
            sound_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                optimizer=tf.optimizers.Adam(),
                                metrics=['accuracy'])

            sound_model.summary()

            # Model study
            EPOCHS = 15

            sound_model.fit(x_train, trainLabels, epochs=EPOCHS, batch_size=2, callbacks=[CustomCallback(), tensorboard])
            QApplication.processEvents()

            # Model prediction
            y_pred = numpy.argmax(sound_model.predict(x_test), axis=1)
            print(y_pred)
            QApplication.processEvents()

            yPred = sound_model.predict(x_test)
            yPred = numpy.argmax(yPred, axis=1)  # Predicted classes
            yTrue = numpy.argmax(testLabels, axis=1)  # True classes

            predPercent = 0.35
            predNumber = int(len(x_test) * predPercent)
            matchNumber = 0

            randomIndices = numpy.random.choice(
                len(x_test),
                predNumber,
                replace=False
            )

            for idx in randomIndices:
                print(f"Sample {idx}: Predicted={yPred[idx]}, Real={yTrue[idx]}")
                if yPred[idx] == yTrue[idx]:
                    matchNumber += 1

            testAcc = (matchNumber / predNumber) * 100

            count = f"Matches: {matchNumber} of {predNumber} is true. Guess Accuracy: {testAcc}"
            print(count)

            # All models saving

            path = "%s/%s, %f" % (savePath, modelType, testAcc)
            saveModel(sound_model, path)

            # Models params for one process identifiaction
            # with open(path + '\params.txt', 'w') as f:
            #     f.write(model_start + "\n")
            #     f.write(model_end)

            resultSet.append(testAcc)
            countSet.append(count)

            # Model params change
            first_layer += 10
            second_layer += 5
            third_layer += 5
            fourth_layer += 3

            # Ui
            QApplication.processEvents()

        return countSet, resultSet
        # 5 models

# Fifth type (FirstType with clustering)
# https://www.tensorflow.org/model_optimization/guide/clustering/clustering_example
@QtCore.pyqtSlot()
def studyFirstTypeClusteringModels():

    if checkStudyData == False:
        consoleOutput.insertPlainText("Відсутній один, або декілька необхідних для навчання наборів даних... \n")
    else:
        modelType = "LSTM_2_DynamicLayers(Clustered)"
        tensorboard = TensorBoard(log_dir=f'TensorboardResults/{modelType}-{int(time.time())}')
        # tensorboard = TensorBoard(log_dir=f'D:/PythonProjects/MagRobota/ModelsResults/{modelType}-{int(time.time())}')

        infoSignal.progress.emit(f'Model: {modelType}, training started!')

        # Start and end params for full study cycle
        # model_start = headers[2]
        # model_start = model_start.split("Гц")[0]
        # model_end = headers[-1]
        # model_end = model_end.split("Гц")[0]

        # Data preparation
        normalize = layers.Normalization()
        normalize.adapt(trainFeatures)

        resultSet = []
        countSet = []

        x_train = numpy.reshape(trainFeatures, (trainFeatures.shape[0], trainFeatures.shape[1],1))
        x_test = numpy.reshape(testFeatures, (testFeatures.shape[0], testFeatures.shape[1],1))

        # Model params
        # last = 100, 75, 50
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

            # Clustering
            infoSignal.progress.emit(f'Clustered layers of: {modelType}, training started!')

            clustered_model = getClusteredModel(sound_model, x_train, trainLabels, x_test, testLabels)

            clustered_model_accuracy = clustered_model.evaluate(
                x_test, testLabels, verbose=0)

            print('Clustered test accuracy:', clustered_model_accuracy)

            # Model prediction (testing)
            yPred = sound_model.predict(x_test)
            yPred = numpy.argmax(yPred, axis=1)  # Predicted classes
            yTrue = numpy.argmax(testLabels, axis=1)  # True classes

            predPercent = 0.35
            predNumber = int(len(x_test) * predPercent)
            matchNumber = 0

            randomIndices = numpy.random.choice(
                len(x_test),
                predNumber,
                replace=False
            )

            for idx in randomIndices:
                print(f"Sample {idx}: Predicted={yPred[idx]}, Real={yTrue[idx]}")
                if yPred[idx] == yTrue[idx]:
                    matchNumber += 1

            testAcc = (matchNumber / predNumber) * 100

            count = f"Matches: {matchNumber} of {predNumber} is true. Guess Accuracy: {testAcc}"
            print(count)

            # All models saving

            path = "%s/%s, %f" % (savePath, modelType, testAcc)

            final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
            saveModel(final_model, path)

            resultSet.append(testAcc)
            countSet.append(count)

            # Model params change
            first_layer += 10
            second_layer += 5

            # Ui
            pBar.setValue(i+1)
            QApplication.processEvents()

        return countSet, resultSet
        # 5 models

# Sixth type (4 layers, squared error, with clustering)
@QtCore.pyqtSlot()
def studySecondTypeClusteringModels():

    if checkStudyData == False:
        consoleOutput.insertPlainText("Відсутній один, або декілька необхідних для навчання наборів даних... \n")
    else:
        modelType = "LSTM_4_Layers(Clustered)"
        tensorboard = TensorBoard(log_dir=f'TensorboardResults/{modelType}-{int(time.time())}')
        # tensorboard = TensorBoard(log_dir=f'D:/PythonProjects/MagRobota/ModelsResults/{modelType}-{int(time.time())}')

        infoSignal.progress.emit(f'Model: {modelType}, training started!')

        # Start and end params for full study cycle
        # model_start = headers[2]
        # model_start = model_start.split("Гц")[0]
        # model_end = headers[-1]
        # model_end = model_end.split("Гц")[0]

        # Data preparation
        normalize = layers.Normalization()
        normalize.adapt(trainFeatures)

        resultSet = []
        countSet = []

        x_train = numpy.reshape(trainFeatures, (trainFeatures.shape[0], trainFeatures.shape[1], 1))
        x_test = numpy.reshape(testFeatures, (testFeatures.shape[0], testFeatures.shape[1], 1))

        # Model params
        first_layer, second_layer, third_layer = 50, 50, 50
        cycles = 5
        for i in range(cycles):
            # Layers
            sound_model = tf.keras.Sequential([
                layers.LSTM(units=first_layer, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                layers.Dropout(0.2),
                layers.LSTM(units=second_layer, return_sequences=True),
                layers.Dropout(0.2),
                layers.LSTM(units=third_layer, return_sequences=True),
                layers.Dropout(0.2),
                layers.LSTM(units=50),
                layers.Dropout(0.2),
                layers.Dense(101),
            ])

            # Model compile
            sound_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                optimizer=tf.optimizers.Adam(),
                                metrics=['accuracy'])
            sound_model.summary()

            # Model study
            EPOCHS = 15

            sound_model.fit(x_train, trainLabels, epochs=EPOCHS, batch_size=2, callbacks=[CustomCallback(), tensorboard])
            QApplication.processEvents()

            # Clustering
            infoSignal.progress.emit(f'Clustered layers of: {modelType}, training started!')

            clustered_model = getClusteredModel(sound_model, x_train, trainLabels, x_test, testLabels)

            clustered_model_accuracy = clustered_model.evaluate(
                x_test, testLabels, verbose=0)

            print('Clustered test accuracy:', clustered_model_accuracy)

            # Model Prediction
            yPred = sound_model.predict(x_test)
            yPred = numpy.argmax(yPred, axis=1)  # Predicted classes
            yTrue = numpy.argmax(testLabels, axis=1)  # True classes

            predPercent = 0.35
            predNumber = int(len(x_test) * predPercent)
            matchNumber = 0

            randomIndices = numpy.random.choice(
                len(x_test),
                predNumber,
                replace=False
            )

            for idx in randomIndices:
                print(f"Sample {idx}: Predicted={yPred[idx]}, Real={yTrue[idx]}")
                if yPred[idx] == yTrue[idx]:
                    matchNumber += 1

            testAcc = (matchNumber / predNumber) * 100

            count = f"Matches: {matchNumber} of {predNumber} is true. Guess Accuracy: {testAcc}"
            print(count)

            # All models saving

            path = "%s/%s, %f" % (savePath, modelType, testAcc)

            final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
            saveModel(final_model, path)

            # Models params for one process identifiaction
            # with open(path + '\params.txt', 'w') as f:
            #     f.write(model_start + "\n")
            #     f.write(model_end)

            resultSet.append(testAcc)
            countSet.append(count)

            # Model params change
            first_layer += 10
            second_layer += 5
            third_layer += 3

            # Ui
            QApplication.processEvents()

        return countSet, resultSet

@QtCore.pyqtSlot()
def studyBirdsModelsFirstType():

    if checkStudyData == False:
        consoleOutput.insertPlainText(
            "Відсутній один, або декілька необхідних для навчання наборів даних... \n")
    else:
        modelType = "Birds (Combined, type 1)"
        tensorboard = TensorBoard(log_dir=f'TensorboardResults/{modelType}-{int(time.time())}')
        # tensorboard = TensorBoard(log_dir=f'D:/PythonProjects/MagRobota/ModelsResults/{modelType}-{int(time.time())}')

        resultSet = []
        countSet = []

        # Data Preparation
        classes = numpy.unique(y_train)

        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )

        class_weights = dict(
            zip(numpy.unique(y_train), weights)
        )

        infoSignal.progress.emit(f'Model: {modelType}, training started!')

        # Model params
        first_layer, second_layer, third_layer = 50, 50, 50
        cycles = 5
        for i in range(cycles):
            # Layers
            model = Sequential([
                layers.Conv1D(32, kernel_size=5, activation='relu', padding='same',
                       input_shape=(X_train.shape[1], X_train.shape[2])),
                keras.layers.BatchNormalization(),
                layers.MaxPooling1D(2),
                layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                layers.MaxPooling1D(2),
                layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.GlobalAveragePooling1D(),
                Dense(128, activation='relu'),
                keras.layers.Dropout(0.4),
                Dense(len(numpy.unique(y_train)), activation='softmax')
            ])

            # Model compile
            model.compile(
                optimizer='Adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            model.summary()

            # Model study
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                shuffle=True,
                callbacks=[early_stop],
                class_weight=class_weights
            )
            QApplication.processEvents()

            # Model Prediction
            le = LabelEncoder()
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

            print(f"Test loss: {test_loss:.4f}")
            print(f"Test accuracy: {test_acc:.4f}")

            y_pred_probs = model.predict(X_test)
            y_pred = y_pred_probs.argmax(axis=1)

            predNumber = 20
            randomPred = show_random_predictions(X_test, y_test, model, le, n=predNumber)

            count = f"Matches: {randomPred} of {predNumber} is true. Guess Accuracy: {test_acc}"
            print(count)

            # All models saving

            if test_acc >= 0.6 and randomPred >= 5:
                path = "%s/%s, %f" % (savePath, modelType, test_acc)
                modelName = model.config["type"] + str(f"_{test_acc:.5f}")
                model.save(f"{path}/{modelName}/birds_model.keras")
                print('Model Saved!')

            resultSet.append(test_acc)
            countSet.append(randomPred)

            # Model params change
            first_layer += 10
            second_layer += 5
            third_layer += 3

            # Ui
            QApplication.processEvents()

        return countSet, resultSet
        # 5 models

# Model manipulation ------------------------------------------------------------------------------------------------

# Clustering model
def getClusteredModel(model, trainFeatures, trainLabels, testFeatures, testLabels):

    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

    clustering_params = {
        'number_of_clusters': 16,
        'cluster_centroids_init': CentroidInitialization.LINEAR
    }

    # Cluster a whole model
    clustered_model = cluster_weights(model, **clustering_params)

    # Use smaller learning rate for fine-tuning clustered model
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

    clustered_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=opt,
        metrics=['accuracy'])

    clustered_model.summary()
    QApplication.processEvents()

    clustered_model.fit(
        trainFeatures, trainLabels,
        batch_size=2,
        epochs=5,
        validation_split=0.1,
        callbacks=[CustomCallback()])
    QApplication.processEvents()

    return clustered_model

# Count correct guesses

# With file names
# def countGuesses(fileNames, y_true, y_pred):
#     if len(y_true) == len(fileNames):
#
#         fileNamesDict = {}
#         namesCount = []
#         names = []
#
#         for n, l in zip(fileNames, y_true):
#             fileNamesDict[n] = l
#
#         for k, v in fileNamesDict.items():
#             before, sep, after = k.rpartition("_")
#
#             names.append(before)
#
#         counter = 0
#
#         for n in names:
#             if y_pred[counter] >= 55 and y_true[counter] == 100:
#                 namesCount.append(n)
#                 counter = counter + 1
#             else:
#                 counter = counter + 1
#
#         print(namesCount)
#
#     values, counts = numpy.unique(namesCount, return_counts=True)
#
#     res = ""
#
#     for i in range(len(values)):
#         res += "File name: %s, %d guesses; " % (values[i], counts[i])
#
#     if res == "":
#         res = "Zero guesses "
#
#     return res

# Without file names
# def countGuesses(y_true, y_pred, cycle):
#     if len(y_true) == len(y_pred):
#
#         counter = 0
#         for i in range(len(y_true)):
#             if y_true[i] == y_pred[i]:
#                 counter += 1
#
#         if counter != 0:
#             res = f"Result: {counter} guesses of {len(y_true)}; "
#         else:
#             res = "Result: Zero guesses "
#
#         return res

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
def useOneModel(loadPath):

    testModel = loadModel(loadPath)

    y_pred = numpy.argmax(testModel.predict(testFeatures), axis=1)

    # y_true = testLabels
    #
    # QApplication.processEvents()
    #
    # rounded = []
    # for v in y_pred:
    #     if v > 55:
    #         rounded.append(100)
    #     else:
    #         rounded.append(0)
    #
    # test_acc = rounded.count(100) / len(y_true)

    y_true = testLabels
    rounded = []

    QApplication.processEvents()

    for v in y_pred:
        if v > 60:
            rounded.append(100)
        else:
            rounded.append(0)

    guesses = 0
    for j in range(len(rounded)):
        if rounded[j] == 100:
            guesses += 1

    test_acc = guesses / len(y_true)

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
def getBestUsable(criteria, loadPath, maxError):

    if criteria == 0:
        # Regularity Criteria
        mse = tf.keras.losses.MeanSquaredError()

        models = os.listdir(loadPath)
        res = []

        for m in models:
            testModel = loadModel(loadPath + "/" + m)

            y_pred = numpy.argmax(testModel.predict(testFeatures), axis=1)
            # y_true = testLabels
            y_true = []
            for i in testLabels:
                y_true.append(i[0])

            print(y_true)
            print(y_pred)
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

def getBestOptimal(criteria, loadPath):

    if criteria == 0:
        # Regularity Criteria
        mse = tf.keras.losses.MeanSquaredError()

        models = os.listdir(loadPath)
        res = {}

        for m in models:
            testModel = loadModel(loadPath + "/" + m)

            y_pred = numpy.argmax(testModel.predict(testFeatures), axis=1)
            # y_true = testLabels
            y_true = []
            for i in testLabels:
                y_true.append(i[0])

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

    elif criteria == 2:
        # Convergence Criteria
        pass

# Models testing ----------------------------------------------------------------------------------------------------

def getAccuracy(loadPath):

    mse = tf.keras.losses.MeanSquaredError()

    model = loadModel(loadPath)

    y_pred = numpy.argmax(model.predict(testFeatures), axis=1)
    # y_true = testLabels
    y_true = []
    for i in testLabels:
        y_true.append(i[0])

    error = mse(y_true, y_pred).numpy()

    error = float(100 - error)

    return f"Точність моделі: {error}"


def getAdequacy(loadPath, trainFeatures, trainLabels, testFeatures, testLabels):
    pass

def getStability(loadPath):

    mse = tf.keras.losses.MeanSquaredError()

    model = loadModel(loadPath)

    res = []

    for i in range(1, 6):

        y_pred = numpy.argmax(model.predict(testFeatures), axis=1)
        # y_true = testLabels
        y_true = []
        for j in testLabels:
            y_true.append(j[0])

        # data alteration (find some method to cahnge array values)
        res.append(f"Цикл {i}, точність: {float(100 - mse(y_true, y_pred).numpy())}")

    return res

def checkStudyData():
    if trainFeatures != 0 and testFeatures != 0 and trainLabels != 0 and testLabels != 0:
        return True
    else:
        return False

def checkTestData():
    if len(testFeatures) != 0 and len(testLabels) != 0:
        print("True")
        return True
    else:
        print("False")
        return False

def show_random_predictions(X, y_true, model, label_encoder, n):
    indices = numpy.random.choice(len(X), n, replace=False)
    res = 0

    for i, idx in enumerate(indices):
        sample = X[idx:idx + 1]

        pred_probs = model.predict(sample, verbose=0)
        pred_class = numpy.argmax(pred_probs)

        true_class = y_true[idx]

        pred_label = label_encoder.inverse_transform([pred_class])[0]
        true_label = label_encoder.inverse_transform([true_class])[0]

        if true_label == pred_label:
            res += 1

        print(f"\nSample {i + 1}")
        print(f"True: {true_label}")
        print(f"Pred: {pred_label}")
        print(f"Confidence: {numpy.max(pred_probs):.3f}")

    print(f"Result: {res}/{n}")
    return res










