import datetime

import os
import librosa
import pandas as pd
import re
import numpy as np
import ast
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.utils import to_categorical

from openpyxl import load_workbook

def loadData():
    # CONFIG
    DATA_PATH = "D:/PythonProjects/MagRobotaData/data/AudioMNIST-master/AudioMNIST-master/data"
    SAMPLE_RATE = 16000
    N_MFCC = 20
    MAX_LEN = 32  # number of time frames for padding
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    def extract_mfcc(file_path):
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=N_MFCC
        )

        # Pad or truncate
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :MAX_LEN]

        return mfcc

    def load_dataset():
        X = []
        y = []

        for speaker in os.listdir(DATA_PATH):
            speaker_path = os.path.join(DATA_PATH, speaker)

            if not os.path.isdir(speaker_path):
                continue

            for file in tqdm(os.listdir(speaker_path), desc=f"Speaker {speaker}"):
                if file.endswith(".wav"):
                    file_path = os.path.join(speaker_path, file)

                    # Extract digit from filename
                    digit = int(file.split("_")[1])

                    mfcc = extract_mfcc(file_path)

                    X.append(mfcc)
                    y.append(digit)

        X = np.array(X)
        y = np.array(y)

        return X, y

    print("Loading and processing dataset...")
    X, y = load_dataset()

    # Normalize
    X = (X - np.mean(X)) / np.std(X)

    # Add channel dimension for CNN
    X = X[..., np.newaxis]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Dataset shapes:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    # Save numerical version
    os.makedirs("D:/PythonProjects/MagRobotaData/processed_dataset", exist_ok=True)

    np.save("D:/PythonProjects/MagRobotaData/processed_dataset/X_train.npy", X_train)
    np.save("D:/PythonProjects/MagRobotaData/processed_dataset/X_test.npy", X_test)
    np.save("D:/PythonProjects/MagRobotaData/processed_dataset/y_train.npy", y_train)
    np.save("D:/PythonProjects/MagRobotaData/processed_dataset/y_test.npy", y_test)

    print("Saved processed dataset!")

def csvToExcel(path, name):
    # Read file line by line
    # Load your CSV (1 column with dicts)
    df_raw = pd.read_csv(path, header=None)

    # Convert each row (string) into dictionary
    dicts = [ast.literal_eval(row[0]) for _, row in df_raw.iterrows()]

    # Convert to proper DataFrame
    df = pd.DataFrame(dicts)

    # Save to Excel
    df.to_excel(name, index=False)

""" Load Other audio dataset
    https://chatgpt.com/c/6988b957-accc-8331-8f9f-e7655d55c41d
"""

# data = loadData()

MNIST_X_train = np.load("D:/PythonProjects/MagRobotaData/processed_dataset/X_train.npy")
MNIST_y_train = np.load("D:/PythonProjects/MagRobotaData/processed_dataset/y_train.npy")
MNIST_X_test = np.load("D:/PythonProjects/MagRobotaData/processed_dataset/X_test.npy")
MNIST_y_test = np.load("D:/PythonProjects/MagRobotaData/processed_dataset/y_test.npy")
# Load the workbook and select the active sheet
wb = load_workbook("D:/PythonProjects/MagRobotaData/data/TestingDump/Test/08.25/Set/20.08_Set.xlsx")  # Replace with your file path
ws = wb.active  # Or wb['SheetName'] if you know the sheet name

# Read all data into a list of lists
data = []
for row in ws.iter_rows(values_only=True):
    # data.append(list(row[1:])) # - Full list
    # data.append(list(row[1:-1])) # - Timbre -1, (3)
    # data.append(list(row[1:-2]))  # - Timbre -2, (2)
    # data.append(list(row[1:-3]))  # - Timbre -3, (1)
    # data.append(list(row[1:-4]))  # - Pitch
    data.append(list(row[1:-5]))  # - Amp only

data = data[1:]
# view = numpy.array(data)

x_data = []
y_data = []

for i in data:
    y_data.append(i[0])
    x_data.append(i[1:])

if __name__ == '__main__':

    """
    MNIST
    """
    runsNumber = 1
    def mnistConv(X_train, y_train, X_test, y_test):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(20, 32, 1)),

            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(61, activation="softmax")
        ])

        for i in range(runsNumber):
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=32,
                validation_data=(X_test, y_test)
            )

            pred_probs = model.predict(X_test)
            y_pred = np.argmax(pred_probs, axis=1)

            # Compare first N samples
            N = 20
            matchNumber = 0

            for i in range(N):
                print(f"Predicted: {y_pred[i]} | Actual: {y_test[i]}")
                if y_pred[i] == y_test[i]:
                    matchNumber += 1

            if (matchNumber / N) * 100 > 80:
                model.save(f"D:/PythonProjects/MagRobotaData/Stattya_SavedModelsForTesting/MNIST_Conv/Model_acc({(matchNumber / N) * 100}),{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    def mnistDense(X_train, y_train, X_test, y_test):
        X_train_dense = X_train.reshape(-1, 20 * 32)
        X_test_dense = X_test.reshape(-1, 20 * 32)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(20 * 32,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(61, activation="softmax")
        ])

        for i in range(runsNumber):
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            model.fit(
                X_train_dense, y_train,
                epochs=10,
                batch_size=32,
                validation_data=(X_test_dense, y_test)
            )

            pred_probs = model.predict(X_test_dense)
            y_pred = np.argmax(pred_probs, axis=1)

            # Compare first N samples
            N = 20
            matchNumber = 0

            for i in range(N):
                print(f"Predicted: {y_pred[i]} | Actual: {y_test[i]}")
                if y_pred[i] == y_test[i]:
                    matchNumber += 1

            if (matchNumber / N) * 100 > 80:
                model.save(
                    f"D:/PythonProjects/MagRobotaData/Stattya_SavedModelsForTesting/MNIST_Dense/Model_acc({(matchNumber / N) * 100}),{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")


    def mnistLSTM(X_train, y_train, X_test, y_test):
        X_train_lstm = X_train.reshape(-1, 32, 20)
        X_test_lstm = X_test.reshape(-1, 32, 20)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(32, 20)),

            tf.keras.layers.LSTM(64, return_sequences=False),

            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(61, activation="softmax")
        ])

        for i in range(runsNumber):
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            model.fit(
                X_train_lstm, y_train,
                epochs=20,
                batch_size=32,
                validation_data=(X_test_lstm, y_test)
            )

            pred_probs = model.predict(X_test_lstm)
            y_pred = np.argmax(pred_probs, axis=1)

            # Compare first N samples
            N = 20
            matchNumber = 0

            for i in range(N):
                print(f"Predicted: {y_pred[i]} | Actual: {y_test[i]}")
                if y_pred[i] == y_test[i]:
                    matchNumber += 1

            if (matchNumber / N) * 100 > 80:
                model.save(
                    f"D:/PythonProjects/MagRobotaData/Stattya_SavedModelsForTesting/MNIST_LSTM/Model_acc({(matchNumber / N) * 100}),{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")


    print("MNIST")
    print("Conv")
    mnistConv(MNIST_X_train, MNIST_y_train, MNIST_X_test, MNIST_y_test)
    print("Dense")
    mnistDense(MNIST_X_train, MNIST_y_train, MNIST_X_test, MNIST_y_test)
    print("LSTM 20 Steps !!!!!!!!")
    mnistLSTM(MNIST_X_train, MNIST_y_train, MNIST_X_test, MNIST_y_test)
    print("over")

    """
    
    CUSTOM DATA ___________________________________________________________________
    
    """

#     """
#     Dense Mlp +-
#     """
#
#     # Simulated input data
#     input_dim = len(x_data[0])
#
#     # X = np.random.rand(num_samples, input_dim).astype(np.float32)
#     # y = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)
#
#     X = np.array(x_data)
#     y = np.array(y_data)
#
#     # Split into train and test sets (80/20)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
#     def build_mlp_model(input_dim):
#         inputs = tf.keras.Input(shape=(input_dim,))
#         x = tf.keras.layers.Dense(256, activation='relu')(inputs)
#         x = tf.keras.layers.BatchNormalization()(x)
#         x = tf.keras.layers.Dense(128, activation='relu')(x)
#         x = tf.keras.layers.Dropout(0.3)(x)
#         x = tf.keras.layers.Dense(64, activation='relu')(x)
#         outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
#         return tf.keras.Model(inputs, outputs)
#
#     empty_df = pd.DataFrame()
#     empty_df.to_excel("mlp_results.xlsx", index=False, header=False)
#
#     numberOftests = 1
#
#     for t in range(1, numberOftests+1):
#
#         mlpModel = build_mlp_model(input_dim)
#
#         mlpModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#         history = mlpModel.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.3)
#
#         # Evaluate on test set
#         test_loss, test_acc = mlpModel.evaluate(X_test, y_test)
#
#         print(f"\nTraining accuracy: {history.history['accuracy'][-1]:.4f}")
#         print(f"\nTraining loss: {history.history['loss'][-1]:.4f}")
#         print(f"\nTraining val accuracy: {history.history['val_accuracy'][-1]:.4f}")
#
#         print(f"\nTest accuracy: {test_acc:.4f}")
#         print(f"\nTest loss: {test_loss:.4f}")
#
#         # Detailed classification report
#         y_pred = mlpModel.predict(X_test)
#         y_pred_classes = (y_pred > 0.5).astype(int)
#
#         matchNumber = 0
#         for i in range(len(y_test)):
#             print(f"{y_test[i]} - {y_pred_classes[i]}")
#             if y_test[i] == y_pred_classes[i][0]:
#                 matchNumber += 1
#         # print(classification_report(y_test, y_pred_classes))
#
#         print(f"Matches: {matchNumber} of {len(y_test)} is true. Percentage Accuracy: {(matchNumber / len(y_test)) * 100}")
#
#         if (matchNumber / len(y_test)) * 100 > 80:
#             mlpModel.save(f"D:/PythonProjects/MagRobotaData/Stattya_SavedModelsForTesting/Mlp/Model_acc({(matchNumber / len(y_test)) * 100}),{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
#
#         results = pd.DataFrame([{
#             "Run": t,
#             "Train Accuracy": history.history['accuracy'][-1],
#             "Train Loss": history.history['loss'][-1],
#             "Val Accuracy": history.history['val_accuracy'][-1],
#             "Test Accuracy": test_acc,
#             "Test Loss": test_loss,
#             f"Test matches of {len(y_test)}": matchNumber,
#             "Test prediction matches %": (matchNumber / len(y_test)) * 100
#         }])
#
#         # ==== Save or append to Excel ====
#         results_df = pd.DataFrame(results)
#
#         excel_file = "mlp_results.xlsx"
#
#
#         try:
#             # Append if file exists
#             existing = pd.read_excel(excel_file)
#             results_df = pd.concat([existing, results_df], ignore_index=True)
#         except FileNotFoundError:
#             # Create new if it doesn’t exist
#             pass
#
#         results_df.to_excel(excel_file, index=False)
#         print(f"Results saved to {excel_file}")
#
#     # -----------------------------------------------------------\
#
#     """
#     Convolutional
#     """
#
#     # Load the workbook and select the active sheet
#     wb = load_workbook(
#         "D:/PythonProjects/MagRobotaData/data/TestingDump/Test/08.25/Set/20.08_SetCombined.xlsx")  # Replace with your file path
#     ws = wb.active  # Or wb['SheetName'] if you know the sheet name
#
#     # Read all data into a list of lists
#     data = []
#     for row in ws.iter_rows(values_only=True):
#         # data.append(list(row[1:])) # - Full list
#         # data.append(list(row[1:-1])) # - Timbre -1, (3)
#         # data.append(list(row[1:-2]))  # - Timbre -2, (2)
#         # data.append(list(row[1:-3]))  # - Timbre -3, (1)
#         # data.append(list(row[1:-4]))  # - Pitch
#         data.append(list(row[1:-5]))  # - Amp only
#
#     data = data[1:]
#
#     x_data = []
#     y_data = []
#
#     for i in data:
#         y_data.append(i[0])
#         x_data.append(i[1:])
#
#     # 3. Normalize features
#     scaler = StandardScaler()
#     X_norm = scaler.fit_transform(x_data)
#
#     # 4. Reshape for CNN: (samples, height=num_frequencies, width=num_features, channels=1)
#     # If each sample has N features (num_columns), treat it as height=N, width=1
#     X_cnn = X_norm.reshape(X_norm.shape[0], X_norm.shape[1], 1, 1)
#
#     # 5. One-hot encode labels
#     y_cat = to_categorical(y_data, num_classes=2)  # adjust num_classes if needed
#
#     # 6. Train/test split
#     X_train, X_test, y_train, y_test = train_test_split(X_cnn, y_cat, test_size=0.3, random_state=42, stratify=y)
#
#     empty_df = pd.DataFrame()
#     empty_df.to_excel("conv_results.xlsx", index=False, header=False)
#
#     for t in range(1, numberOftests+1):
#
#         convModel = Sequential([
#             Conv2D(16, (3, 1), activation='relu', padding='same', input_shape=(X_cnn.shape[1], 1, 1)),
#             MaxPooling2D((2, 1)),
#             Dropout(0.3),
#
#             Conv2D(32, (3, 1), activation='relu', padding='same'),
#             MaxPooling2D((2, 1)),
#             Dropout(0.3),
#
#             Flatten(),
#             Dense(64, activation='relu'),
#             Dropout(0.4),
#             Dense(2, activation='softmax')
#         ])
#
#         convModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         convModel.summary()
#
#         # 8. Train
#         history = convModel.fit(X_train, y_train, validation_split=0.3, epochs=10, batch_size=32)
#
#         # 9. Evaluate
#         test_loss, test_acc = convModel.evaluate(X_test, y_test)
#         print(f"Test Accuracy: {test_acc:.4f}")
#         # Predictions
#         y_pred = convModel.predict(X_test)
#         y_pred_classes = (y_pred > 0.5).astype(int)
#
#         for i in range(len(y_pred)):
#             print(f"{y_pred[i]} - {y_test[i]}")
#
#         test_loss, test_acc = convModel.evaluate(X_test, y_test)
#         print(f"Test Accuracy: {test_acc:.4f}")
#
#         # 10. Predictions
#         y_pred_probs = convModel.predict(X_test)  # Probabilities
#         y_pred = np.argmax(y_pred_probs, axis=1)  # Predicted classes
#         y_true = np.argmax(y_test, axis=1)  # True classes
#
#         # Show first 20 predictions vs real values
#         pred_number = 23
#         matchNumber = 0
#         for i in range(pred_number):
#             print(f"Sample {i}: Predicted={y_pred[i]}, Real={y_true[i]}")
#             if y_pred[i] == y_true[i]:
#                 matchNumber += 1
#         print(f"Prediction Accuracy: {matchNumber} of {pred_number} is true. Percentage Accuracy: {(matchNumber/pred_number) * 100}")
#
#         if (matchNumber/pred_number) * 100 > 80:
#             convModel.save(f"D:/PythonProjects/MagRobotaData/Stattya_SavedModelsForTesting/Convolutional//Model_acc({(matchNumber/pred_number) * 100}), {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
#
#         results = pd.DataFrame([{
#             "Run": t,
#             "Train Accuracy": history.history['accuracy'][-1],
#             "Train Loss": history.history['loss'][-1],
#             "Val Accuracy": history.history['val_accuracy'][-1],
#             "Test Accuracy": test_acc,
#             "Test Loss": test_loss,
#             f"Test matches of {pred_number}": matchNumber,
#             "Test prediction matches %": (matchNumber / pred_number) * 100
#         }])
#
#         # ==== Save or append to Excel ====
#         results_df = pd.DataFrame(results)
#
#         excel_file = "conv_results.xlsx"
#
#         try:
#             # Append if file exists
#             existing = pd.read_excel(excel_file)
#             results_df = pd.concat([existing, results_df], ignore_index=True)
#         except FileNotFoundError:
#             # Create new if it doesn’t exist
#             pass
#
#         results_df.to_excel(excel_file, index=False)
#         print(f"Results saved to {excel_file}")
#
#     # -------------------------------------------------------------
#
#     """
#     LSTM via 2D, excluding timestamps
#     """
#
#     wb = load_workbook("D:/PythonProjects/MagRobotaData/data/TestingDump/Test/08.25/Set/20.08_SetCombined.xlsx")
#     ws = wb.active
#
#     data = []
#     for row in ws.iter_rows(values_only=True):
#         # data.append(list(row[1:])) # - Full list
#         # data.append(list(row[1:-1])) # - Timbre -1, (3)
#         # data.append(list(row[1:-2]))  # - Timbre -2, (2)
#         # data.append(list(row[1:-3]))  # - Timbre -3, (1)
#         # data.append(list(row[1:-4]))  # - Pitch
#         data.append(list(row[1:-5]))  # - Amp only
#
#     data = data[1:]  # skip header
#
#     x_data = []
#     y_data = []
#
#     for row in data:
#         y_data.append(row[0])  # label in column B
#         x_data.append(row[1:])  # all numeric features
#
#     x_data = np.array(x_data, dtype=float)
#     y_data = np.array(y_data, dtype=int)
#
#     # 2. Normalize features
#     scaler = StandardScaler()
#     X_norm = scaler.fit_transform(x_data)
#
#     # 3. Reshape for LSTM: (samples, timesteps, features)
#     # Here timesteps = number of feature columns, features = 1
#     X_lstm = X_norm.reshape(X_norm.shape[0], X_norm.shape[1], 1)
#
#     # 4. One-hot encode labels
#     y_cat = to_categorical(y_data, num_classes=2)
#
#     # 5. Train/test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_lstm, y_cat, test_size=0.3, random_state=42, stratify=y_data
#     )
#
#     print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
#
#     # 6. LSTM Model
#
#     empty_df = pd.DataFrame()
#     empty_df.to_excel("lstm2D_results.xlsx", index=False, header=False)
#
#     for t in range(1, numberOftests + 1):
#
#         lstm2DModel = Sequential([
#             LSTM(64, input_shape=(X_lstm.shape[1], 1), return_sequences=True),
#             Dropout(0.3),
#             LSTM(32),
#             Dropout(0.3),
#             Dense(64, activation='relu'),
#             Dropout(0.4),
#             Dense(2, activation='softmax')
#         ])
#
#         lstm2DModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         lstm2DModel.summary()
#
#         # 7. Train
#         history = lstm2DModel.fit(X_train, y_train, validation_split=0.3, epochs=15, batch_size=32)
#
#         # 8. Evaluate
#         test_loss, test_acc = lstm2DModel.evaluate(X_test, y_test)
#         print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
#
#         # 9. Predictions
#         y_pred_probs = lstm2DModel.predict(X_test)
#         y_pred = np.argmax(y_pred_probs, axis=1)
#         y_true = np.argmax(y_test, axis=1)
#
#         pred_number = 23
#         matchNumber = 0
#         for i in range(pred_number):
#             print(f"Sample {i}: Predicted={y_pred[i]}, Real={y_true[i]}")
#             if y_pred[i] == y_true[i]:
#                 matchNumber += 1
#         print(
#             f"Prediction Accuracy: {matchNumber} of {pred_number} is true. Percentage Accuracy: {(matchNumber / pred_number) * 100}")
#
#         if (matchNumber/pred_number) * 100 > 80:
#             lstm2DModel.save(f"D:/PythonProjects/MagRobotaData/Stattya_SavedModelsForTesting/LSTM2/Model_acc({(matchNumber / pred_number) * 100}), {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
#
#         results = pd.DataFrame([{
#             "Run": t,
#             "Train Accuracy": history.history['accuracy'][-1],
#             "Train Loss": history.history['loss'][-1],
#             "Val Accuracy": history.history['val_accuracy'][-1],
#             "Test Accuracy": test_acc,
#             "Test Loss": test_loss,
#             f"Test matches of {pred_number}": matchNumber,
#             "Test prediction matches %": (matchNumber / pred_number) * 100
#         }])
#
#         # ==== Save or append to Excel ====
#         results_df = pd.DataFrame(results)
#
#         excel_file = "lstm2D_results.xlsx"
#
#         try:
#             # Append if file exists
#             existing = pd.read_excel(excel_file)
#             results_df = pd.concat([existing, results_df], ignore_index=True)
#         except FileNotFoundError:
#             # Create new if it doesn’t exist
#             pass
#
#         results_df.to_excel(excel_file, index=False)
#         print(f"Results saved to {excel_file}")
#
# # -----------------------------------------------------------------------------
#
# # -----------------------------------------------------------------------------
#
#     """
#     LSTM - Via sequences
#     """
#
#     # Load the workbook and select the active sheet
#     wb = load_workbook("D:/PythonProjects/MagRobotaData/data/TestingDump/Test/08.25/Set/20.08_SetCombined.xlsx")
#     ws = wb.active
#
#     data = []
#     for row in ws.iter_rows(values_only=True):
#         data.append(list(row))
#
#     data = data[1:]
#
#     file_ids = [row[0] for row in data]  # col A
#     y_data = [row[1] for row in data]  # col B
#     # x_data = [row[2:] for row in data]  # cols C..end
#     # x_data = [row[2:-1] for row in data]
#     # x_data = [row[2:-2] for row in data]
#     # x_data = [row[2:-3] for row in data]
#     # x_data = [row[2:-4] for row in data]
#     x_data = [row[2:-5] for row in data]
#
#     # data.append(list(row[1:])) # - Full list
#     # data.append(list(row[1:-1])) # - Timbre -1, (3)
#     # data.append(list(row[1:-2]))  # - Timbre -2, (2)
#     # data.append(list(row[1:-3]))  # - Timbre -3, (1)
#     # data.append(list(row[1:-4]))  # - Pitch
#     # data.append(list(row[1:-5]))  # - Amp only
#
#     x_data = np.array(x_data, dtype=float)
#     y_data = np.array(y_data, dtype=int)
#
#     # Normalize features
#     scaler = StandardScaler()
#     x_data = scaler.fit_transform(x_data)
#
#     def get_base_id(filename):
#         # Example: "file_a_12.wav" → "file_a"
#         return re.sub(r"_\d+\.wav$", "", filename)
#
#     base_ids = [get_base_id(row[0]) for row in data]
#
#     # Parameters
#     SEQ_LEN = 5  # number of timesteps per sequence
#     features = x_data.shape[1]
#
#     X_seq, y_seq = [], []
#
#     unique_files = list(set(base_ids))
#     for f in unique_files:
#         indices = [i for i, fid in enumerate(base_ids) if fid == f]
#         seq_features = x_data[indices]
#         seq_labels = y_data[indices]
#
#         # slice into sequences
#         for i in range(0, len(indices) - SEQ_LEN + 1, SEQ_LEN):
#             X_seq.append(seq_features[i:i + SEQ_LEN])
#             y_seq.append(int(np.round(np.mean(seq_labels[i:i + SEQ_LEN]))))
#
#     X_seq = np.array(X_seq)
#     y_seq = np.array(y_seq)
#
#     print("Final shape:", X_seq.shape, y_seq.shape)  # (num_sequences, SEQ_LEN, features)
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_seq, y_seq, test_size=0.3, random_state=42, stratify=y_seq
#     )
#
#     y_train = to_categorical(y_train, num_classes=2)
#     y_test = to_categorical(y_test, num_classes=2)
#
#     print("Train:", X_train.shape, y_train.shape)
#     print("Test:", X_test.shape, y_test.shape)
#
#     empty_df = pd.DataFrame()
#     empty_df.to_excel("lstm3D_results.xlsx", index=False, header=False)
#
#     for t in range(1, numberOftests + 1):
#
#         lstm3Dmodel = Sequential([
#             LSTM(128, input_shape=(X_seq.shape[1], X_seq.shape[2]), return_sequences=True),
#             Dropout(0.3),
#             LSTM(64),
#             Dropout(0.3),
#             Dense(64, activation='relu'),
#             Dropout(0.4),
#             Dense(2, activation='softmax')
#         ])
#
#         lstm3Dmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         lstm3Dmodel.summary()
#
#         history = lstm3Dmodel.fit(
#             X_train, y_train,
#             validation_split=0.2,
#             epochs=30,
#             batch_size=8,
#             verbose=1
#         )
#
#         test_loss, test_acc = lstm3Dmodel.evaluate(X_test, y_test)
#         print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
#
#         # Predictions
#         y_pred_probs = lstm3Dmodel.predict(X_test)
#         y_pred = np.argmax(y_pred_probs, axis=1)
#         y_true = np.argmax(y_test, axis=1)
#         # y_true = np.argmax(y_test)
#
#         print(y_pred)
#         print(y_true)
#         matchNumber = 0
#         for i in range(len(y_true)):
#             if y_pred[i] == y_true[i]:
#                 matchNumber += 1
#             print(f"Seq {i}: Pred={y_pred[i]} ({y_pred_probs[i][y_pred[i]] * 100:.1f}%) | Real={y_true[i]}")
#
#         if test_acc * 100 > 50:
#             lstm3Dmodel.save(f"D:/PythonProjects/MagRobotaData/Stattya_SavedModelsForTesting/LSTM3/Model_acc({test_acc * 100}), {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
#
#         results = pd.DataFrame([{
#             "Run": t,
#             "Train Accuracy": history.history['accuracy'][-1],
#             "Train Loss": history.history['loss'][-1],
#             "Val Accuracy": history.history['val_accuracy'][-1],
#             "Test Accuracy": test_acc,
#             "Test Loss": test_loss,
#             f"Test matches of {len(y_pred)}": matchNumber,
#             "Test prediction matches %": (matchNumber / len(y_true)) * 100
#         }])
#
#         # ==== Save or append to Excel ====
#         results_df = pd.DataFrame(results)
#
#         excel_file = "lstm3D_results.xlsx"
#
#         try:
#             # Append if file exists
#             existing = pd.read_excel(excel_file)
#             results_df = pd.concat([existing, results_df], ignore_index=True)
#         except FileNotFoundError:
#             # Create new if it doesn’t exist
#             pass
#
#         results_df.to_excel(excel_file, index=False)
#         print(f"Results saved to {excel_file}")




