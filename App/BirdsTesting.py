import os

import keras.layers
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from collections import Counter
import pickle
import json

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, Dense, LSTM
from keras import regularizers, callbacks
import BirdsModels

if __name__ == '__main__':
    # =========================
    # CONFIG
    # =========================

    # Max frequency = sample_rate / 2
    # sr = 22050 → max freq ≈ 11 kHz
    # sr = 16000 → max freq ≈ 8 kHz
    # sr = 8000 → max freq ≈ 4 kHz

    # DATASET_PATH = "D:\PythonProjects\AudioBirds"
    DATASET_PATH = "D:\PythonProjects\AudioBirds_Short"

    # For spectrogram

    # SAMPLE_RATE = 16000
    # CHUNK_DURATION = 3.0  # seconds
    # CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
    # MAX_FILES_PER_CLASS = 40

    # For AudioData

    SAMPLE_RATE = 22050
    # SAMPLE_RATE = 16000
    # SAMPLE_RATE = 8000
    CHUNK_DURATION = 3.0  # seconds
    # CHUNK_DURATION = 1.5  # seconds
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
    MAX_FILES_PER_CLASS = 20

    # =========================
    # PLOTS
    # =========================

    def plot_grid(X, y, num=15):
        plt.figure(figsize=(12, 8))

        for i in range(num):
            plt.subplot(num, 1, i + 1)
            plt.plot(X[i].squeeze())
            plt.title(f"Label: {y[i]}")

        plt.tight_layout()
        plt.show()


    def plot_spectrogram_grid(X, y, label_encoder=None, rows=3, cols=3):

        total = rows * cols

        plt.figure(figsize=(15, 10))

        indices = np.random.choice(len(X), total, replace=False)

        for i, idx in enumerate(indices):

            plt.subplot(rows, cols, i + 1)

            # remove channel dimension
            mel = X[idx].squeeze()

            plt.imshow(
                mel,
                aspect='auto',
                origin='lower'
            )

            # decode label if encoder exists
            if label_encoder is not None:
                label = label_encoder.inverse_transform([y[idx]])[0]
            else:
                label = y[idx]

            plt.title(f"Label: {label}")

            plt.xlabel("Time")
            plt.ylabel("Mel")

        plt.tight_layout()
        plt.show()

    def inspect_samples(X, y, num=15):
        for i in range(num):
            print(f"Sample {i} | Label: {y[i]}")
            print(X[i].squeeze()[:20])  # first values only

            plt.figure(figsize=(8, 2))
            plt.plot(X[i].squeeze())
            plt.show()

    def plot_mel(mel):
        plt.figure(figsize=(6, 4))
        plt.imshow(mel, aspect='auto', origin='lower')
        plt.title("Mel Spectrogram")
        plt.xlabel("Time")
        plt.ylabel("Mel bins")
        plt.colorbar()
        plt.show()

    def chunks_to_dataframe(X, y, max_points=200):
        rows = []

        for i in range(len(X)):
            chunk = X[i].squeeze()

            # optionally downsample for readability
            if len(chunk) > max_points:
                step = len(chunk) // max_points
                chunk = chunk[::step]

            row = {"label": y[i]}

            for j, val in enumerate(chunk):
                row[f"x{j}"] = val

            rows.append(row)

        return pd.DataFrame(rows)


    def export_to_excel(X, y_labels, label_encoder, filename="birds.xlsx", max_points=500):
        rows = []

        for i in range(len(X)):
            chunk = X[i].squeeze()

            # Optional: downsample for readability
            if len(chunk) > max_points:
                step = len(chunk) // max_points
                chunk = chunk[::step]

            row = {
                "bird_name": label_encoder.inverse_transform([y_labels[i]])[0],
                "class_id": int(y_labels[i]),
                "amp_data": chunk.tolist()
            }

            rows.append(row)

        df = pd.DataFrame(rows)

        df.to_excel(filename, index=False)

        print(f"Saved to {filename}")

    import pandas as pd
    import numpy as np


    # =========================
    # NORMALIZATION
    # =========================
    def rms_normalize(y, target_rms=0.1):
        rms = np.sqrt(np.mean(y**2))
        if rms == 0:
            return y
        return y * (target_rms / rms)

    # =========================
    # FIXED SIZE HANDLING
    # =========================
    def fix_length(chunk, size):
        if len(chunk) < size:
            # pad with zeros
            return np.pad(chunk, (0, size - len(chunk)))
        else:
            return chunk[:size]

    # =========================
    # SPLIT INTO CHUNKS
    # =========================
    def split_into_chunks(y, chunk_size):
        chunks = []
        for i in range(0, len(y), chunk_size):
            chunk = y[i:i + chunk_size]
            if len(chunk) < chunk_size:
                continue  # skip small tail (or pad if you prefer)
            chunks.append(chunk)
        return chunks


    # =========================
    # PROCESS SINGLE FILE
    # =========================
    # def chunk_to_mel(chunk, sr=22050, n_mels=128, n_fft=1024, hop_length=512):
    #     # Mel spectrogram
    #     mel = librosa.feature.melspectrogram(
    #         y=chunk,
    #         sr=sr,
    #         n_mels=n_mels,
    #         n_fft=n_fft,
    #         hop_length=hop_length,
    #         fmin=1000,
    #         fmax=10000
    #     )
    #
    #     # Convert to log scale (VERY important)
    #     mel_db = librosa.power_to_db(mel, ref=np.max)
    #
    #     # Normalize
    #     mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)
    #
    #     return mel_db


    def fix_spectrogram_size(mel, target_width=130):
        current_width = mel.shape[1]

        if current_width < target_width:
            # pad with zeros
            pad_width = target_width - current_width
            mel = np.pad(mel, ((0, 0), (0, pad_width)))
        else:
            # crop
            mel = mel[:, :target_width]

        return mel

    def process_file(file_path):
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Normalize
        y = rms_normalize(y)

        # Split
        # chunks = split_into_chunks(y, CHUNK_SIZE, 1)
        chunks = split_into_chunks(y, CHUNK_SIZE)

        # Ensure fixed size
        chunks = [fix_length(c, CHUNK_SIZE) for c in chunks]

        return chunks

    def process_file_spectrogram(file_path, augment=False):
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        y = rms_normalize(y)
        chunks = split_into_chunks(y, CHUNK_SIZE)
        # chunks = split_into_chunks(y, CHUNK_SIZE, 1)

        processed = []

        for chunk in chunks:

            if len(chunk) < CHUNK_SIZE:
                continue

            # if np.mean(np.abs(chunk)) < 0.01:
            #     continue

            energy = np.mean(librosa.feature.rms(y=chunk))
            if energy < 0.02:
                continue

            # =========================
            # AUGMENTATION (TRAIN ONLY)
            # =========================
            if augment:
                # random noise
                if np.random.rand() < 0.5:
                    noise = np.random.randn(len(chunk))
                    chunk = chunk + 0.005 * noise

                # random shift
                if np.random.rand() < 0.5:
                    shift = int(np.random.uniform(-0.2, 0.2) * len(chunk))
                    chunk = np.roll(chunk, shift)

            chunk = librosa.effects.preemphasis(chunk)

            mel = librosa.feature.melspectrogram(
                y=chunk,
                sr=sr,
                n_mels=128,
                n_fft=1024,
                hop_length=512,
                fmin=2500,
                fmax=8000
            )

            # Convert to log scale (VERY important)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # =========================
            # BIRD ACTIVITY FILTER
            # =========================
            energy = np.mean(mel_db)
            if energy < -60:
                continue

            mel = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)

            # mel = fix_spectrogram_size(mel, target_width=130)

            # Add channel dimension (for CNN)
            mel = mel[..., np.newaxis]

            processed.append(mel)

        return processed

    def process_file_multifeature(file_path):
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Normalize
        y = rms_normalize(y)

        # Split into chunks
        chunks = split_into_chunks(y, CHUNK_SIZE)
        # chunks = split_into_chunks(y, CHUNK_SIZE, 1)

        processed_chunks = []

        for chunk in chunks:
            chunk = fix_length(chunk, CHUNK_SIZE)

            # =========================
            # RMS & ZCR (frame-based)
            # =========================
            rms = librosa.feature.rms(y=chunk)[0]
            zcr = librosa.feature.zero_crossing_rate(chunk)[0]

            # =========================
            # Resize to match chunk length
            # =========================
            def expand_feature(feature, target_length):
                return np.interp(
                    np.linspace(0, len(feature), num=target_length),
                    np.arange(len(feature)),
                    feature
                )

            rms_expanded = expand_feature(rms, len(chunk))
            zcr_expanded = expand_feature(zcr, len(chunk))

            # =========================
            # Stack channels
            # =========================
            combined = np.stack([
                normalize_feature(chunk),
                normalize_feature(rms_expanded),
                normalize_feature(zcr_expanded)
            ], axis=-1)

            processed_chunks.append(combined)

        return processed_chunks


    def normalize_feature(f):
        return (f - np.mean(f)) / (np.std(f) + 1e-8)


    def downsample_sequence(data, factor=10):
        # data shape: (samples, time, features)
        samples, time_steps, feat = data.shape
        new_length = time_steps // factor
        # Trim to a multiple of factor
        trimmed = data[:, :new_length * factor, :]
        # Reshape and mean over the factor dimension
        return trimmed.reshape(samples, new_length, factor, feat).mean(axis=2)
    # =========================
    # PROCESS DATASET
    # =========================

    '''Initial'''

    # def process_dataset(dataset_path):
    #     X = []
    #     y = []
    #
    #     labels = []
    #
    #     for label in os.listdir(dataset_path):
    #         folder_path = os.path.join(dataset_path, label)
    #
    #         if not os.path.isdir(folder_path):
    #             continue
    #
    #         labels.append(label)
    #
    #         files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    #
    #         if MAX_FILES_PER_CLASS:
    #             files = files[:MAX_FILES_PER_CLASS]
    #
    #         print(f"\nProcessing class: {label} ({len(files)} files)")
    #
    #         for file in tqdm(files):
    #             full_path = os.path.join(folder_path, file)
    #
    #             try:
    #
    #                 chunks = process_file(full_path)
    #
    #                 for chunk in chunks:
    #                     X.append(chunk)
    #                     y.append(label)
    #
    #             except Exception as e:
    #                 print(f"Error processing {file}: {e}")
    #
    #     return np.array(X), np.array(y)


    """
    Main Procession
    Data type (Sound, Spectrogram etc) is chosen after try: cycle starts
    """
    def process_dataset_split(dataset_path, test_size=0.2):
        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []

        for label in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, label)

            if not os.path.isdir(folder_path):
                continue

            files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

            if MAX_FILES_PER_CLASS:
                files = files[:MAX_FILES_PER_CLASS]

            train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)
            val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

            print(f"\nClass: {label}")
            print(f"Train files: {len(train_files)}, Val files: {len(val_files)}, , Test files: {len(test_files)}")

            for file in tqdm(train_files, desc=f"{label} train"):
                full_path = os.path.join(folder_path, file)

                try:
                    # chunks = process_file(full_path)
                    chunks = process_file_multifeature(full_path)
                    # chunks = process_file_spectrogram(full_path, augment=True)

                    for chunk in chunks:
                        X_train.append(chunk)
                        y_train.append(label)

                except Exception as e:
                    print(f"Error processing {file}: {e}")

            for file in tqdm(val_files, desc=f"{label} val"):
                full_path = os.path.join(folder_path, file)

                try:
                    # chunks = process_file(full_path)
                    chunks = process_file_multifeature(full_path)
                    # chunks = process_file_spectrogram(full_path, augment=False)

                    for chunk in chunks:
                        X_val.append(chunk)
                        y_val.append(label)

                except Exception as e:
                    print(f"Error processing {file}: {e}")

            for file in tqdm(test_files, desc=f"{label} test"):
                full_path = os.path.join(folder_path, file)

                try:
                    # chunks = process_file(full_path)
                    chunks = process_file_multifeature(full_path)
                    # chunks = process_file_spectrogram(full_path, augment=False)

                    for chunk in chunks:
                        X_test.append(chunk)
                        y_test.append(label)

                except Exception as e:
                    print(f"Error processing {file}: {e}")

        return (
            np.array(X_train), np.array(y_train),
            np.array(X_val), np.array(y_val),
            np.array(X_test), np.array(y_test)
        )

    # =========================
    # RUN PIPELINE
    # =========================

    # For audio Data
    # X, y = process_dataset(DATASET_PATH)
    # X_train = X_train / (np.max(np.abs(X_train), axis=1, keepdims=True) + 1e-8)
    # X_val = X_val / (np.max(np.abs(X_val), axis=1, keepdims=True) + 1e-8)
    # X_test = X_test / (np.max(np.abs(X_test), axis=1, keepdims=True) + 1e-8)


    # For spectrogram
    X_train, y_train, X_val, y_val, X_test, y_test = process_dataset_split(DATASET_PATH)
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    X_train_ds = downsample_sequence(X_train, factor=10)  # (665, 6615, 3)
    X_val_ds = downsample_sequence(X_val, factor=10)
    X_test_ds = downsample_sequence(X_test, factor=10)

    # Encode labels
    le = LabelEncoder()
    # y_encoded = le.fit_transform(y)

    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)

    print(len(np.unique(y_train)))

    # plot_spectrogram_grid(
    #     X_train,
    #     y_train,
    #     label_encoder=le,
    #     rows=3,
    #     cols=3
    # )

    # export_to_excel(X_train, y_train, le, "train_data.xlsx")
    # export_to_excel(X_val, y_val, le, "validation_data.xlsx")
    # export_to_excel(X_test, y_test, le, "test_data.xlsx")
    # export_full_chunks(X_val, y_val, le, "validation_data_full.xlsx")

    # =========================
    # FINAL SHAPE FOR 1D CNN
    # =========================
    # (samples, timesteps, channels)

    # X_train = X_train[..., np.newaxis]
    # X_val = X_val[..., np.newaxis]
    # X_test = X_test[..., np.newaxis]

    # =========================
    # SAVE
    # =========================
    # np.save("X_train.npy", X_train)
    # np.save("X_val.npy", X_val)
    # np.save("X_test.npy", X_test)
    # np.save("y_train.npy", y_train)
    # np.save("y_test.npy", y_test)
    # np.save("y_val.npy", y_val)

    # =========================
    # TEST
    # =========================

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y_encoded,
    #     test_size=0.2,
    #     random_state=42,
    #     stratify=y_encoded
    # )
    #
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_train, y_train,
    #     test_size=0.2,
    #     random_state=42,
    #     stratify=y_train
    # )

    def show_random_predictions(X, y_true, model, label_encoder, n):
        indices = np.random.choice(len(X), n, replace=False)
        res = 0

        for i, idx in enumerate(indices):
            sample = X[idx:idx + 1]

            pred_probs = model.predict(sample, verbose=0)
            pred_class = np.argmax(pred_probs)

            true_class = y_true[idx]

            pred_label = label_encoder.inverse_transform([pred_class])[0]
            true_label = label_encoder.inverse_transform([true_class])[0]

            # plt.figure(figsize=(10, 3))
            # plt.plot(X[idx].squeeze())
            #
            # plt.title(f"True: {true_label} | Pred: {pred_label} | Conf: {np.max(pred_probs):.2f}")
            # plt.show()

            if true_label == pred_label:
                res += 1

            print(f"\nSample {i + 1}")
            print(f"True: {true_label}")
            print(f"Pred: {pred_label}")
            print(f"Confidence: {np.max(pred_probs):.3f}")

        print(f"Result: {res}/{n}")
        return res

    # model = Sequential([
    #     Conv1D(16, 5, activation='relu', input_shape=(CHUNK_SIZE, 1), kernel_regularizer=regularizers.l2(0.001)),
    #     # Conv1D(32, 5, activation='relu', input_shape=(CHUNK_SIZE, 1), kernel_regularizer=regularizers.l2(0.001)),
    #     MaxPooling1D(2),
    #     keras.layers.Dropout(0.3),
    #     Conv1D(32, 5, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    #     # Conv1D(64, 5, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    #     MaxPooling1D(2),
    #     keras.layers.Dropout(0.2),
    #     Flatten(),
    #     Dense(64, activation='relu'),
    #     Dense(len(np.unique(y_train)), activation='softmax')
    # ])

    """
    70% on 30 epochs
    - Amp
    - RMS
    - ZCR
    """

    modelConv70 = Sequential([
        Conv1D(64, 5, activation='relu', input_shape=(CHUNK_SIZE, 3)),
        MaxPooling1D(2),
        Conv1D(32, 5, activation='relu'),
        MaxPooling1D(2),
        keras.layers.Dropout(0.2),
        keras.layers.GlobalAveragePooling1D(),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])

    """Spectrogram"""

    # modelSpectrogram = Sequential([
    #     Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(64, (3, 3), activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPooling2D((2, 2)),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     keras.layers.Dropout(0.3),
    #     Dense(len(np.unique(y_train)), activation='softmax')
    # ])

    """
    Good?
    """

    # model = Sequential([
    #     Conv2D(32, (3, 3), activation='relu',
    #            input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    #     keras.layers.BatchNormalization(),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(64, (3, 3), activation='relu'),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPooling2D((2, 2)),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     keras.layers.Dropout(0.5),
    #     Dense(len(np.unique(y_train)), activation='softmax')
    # ])

    modelConv = Sequential([
        # Conv1D(32, 5, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(CHUNK_SIZE, 1)),
        Conv1D(32, 5, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(CHUNK_SIZE, 3)),
        MaxPooling1D(2),
        keras.layers.Dropout(0.1),
        # Block 2
        Conv1D(64, 5, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        MaxPooling1D(2),
        keras.layers.Dropout(0.1),
        # Block 3 (optional deeper feature extraction)
        Conv1D(128, 3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        MaxPooling1D(2),
        keras.layers.Dropout(0.2),
        # Classifier head
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])

    """
    70% on 10-9 epochs - Good one
    - Amp
    - RMS
    - ZCR
    """

    modelSeveralDenseBlocks = Sequential([
        keras.layers.Input(shape=(CHUNK_SIZE, 3)),
        # First dense block
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        # Second dense block
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        # Third dense block
        keras.layers.Dense(16, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.GlobalAveragePooling1D(),
        # Output layer
        keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
    ])

    """
    70% on 5 epochs - Good one
    - Amp
    - RMS
    - ZCR
    """

    model = BirdsModels.build_model((CHUNK_SIZE, 3), len(np.unique(y_train)))

    model.compile(
        optimizer='Adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    model.config = {
        "type": "spectrogram",
    }

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        shuffle=True,
        callbacks=[early_stop]
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = y_pred_probs.argmax(axis=1)

    randomPred = show_random_predictions(X_test, y_test, model, le, n=10)

    if test_acc >= 0.6 and randomPred >= 5:
        modelName = model.config["type"] + str(f"_{test_acc:.5f}")
        model.save(f"D:/PythonProjects/MagRobotaData/BirdsData/Models/{modelName}/birds_model.keras")

        with open(f"D:/PythonProjects/MagRobotaData/BirdsData/Models/{modelName}/label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)
            config = {
            "sample_rate": SAMPLE_RATE,
            "chunk_size": CHUNK_SIZE,
            "channels": 3
        }

        with open(f"D:/PythonProjects/MagRobotaData/BirdsData/Models/{modelName}/config.json", "w") as f:
            json.dump(config, f)

        print('Model Saved!')


    # test_loss, test_acc = model.evaluate(X_test, y_test)
    # print("Test accuracy:", test_acc)
    #
    # y_pred = model.predict(X_test)
    # y_pred_classes = y_pred.argmax(axis=1)
    #
    # print(y_pred)
    #
    # print(classification_report(y_test, y_pred_classes))




