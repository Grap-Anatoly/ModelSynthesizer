import os

import tensorflow as tf
import keras.layers
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import pickle
import json

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, Dense, LSTM
from keras import regularizers, callbacks
import BirdsModels

"""

- To Do: Feature importance, new data sets one by one without one feature x-y=i (importance)
- Main cycle - data transformation
- Multiple models for different data in model types 

"""


if __name__ == '__main__':

    keras.backend.clear_session()

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
    CHUNK_DURATION = 3.0  # seconds
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
    # 20 is default
    MAX_FILES_PER_CLASS = 50
    TARGET_CHUNKS = 150

    """PLOTS"""

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

    """
    Export
    """

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

    """
    Util
    """
    def rms_normalize(y, target_rms=0.1):
        rms = np.sqrt(np.mean(y**2))
        if rms == 0:
            return y
        return y * (target_rms / rms)

    def fix_length(chunk, size):
        if len(chunk) < size:
            # pad with zeros
            return np.pad(chunk, (0, size - len(chunk)))
        else:
            return chunk[:size]

    def split_into_chunks(y, chunk_size):
        chunks = []
        for i in range(0, len(y), chunk_size):
            chunk = y[i:i + chunk_size]
            if len(chunk) < chunk_size:
                continue  # skip small tail (or pad if you prefer)
            chunks.append(chunk)
        return chunks

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

    """
    Process Files
    """
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

        """
        Resize to match chunk length
        """
        def expand_feature(feature, target_length):
            return np.interp(
                np.linspace(0, len(feature), num=target_length),
                np.arange(len(feature)),
                feature
            )

        processed_chunks = []
        """
        Noise detection by RMS percentile
        """
        chunk_rms_values = []

        for chunk in chunks:
            chunk = fix_length(chunk, CHUNK_SIZE)

            rms = librosa.feature.rms(y=chunk)[0]
            chunk_rms_values.append(np.mean(rms))

        activity_threshold = np.percentile(
            chunk_rms_values,
            40
        )

        for chunk in chunks:
            chunk = fix_length(chunk, CHUNK_SIZE)

            """
            RMS and ZCR
            """
            rms = librosa.feature.rms(y=chunk)[0]

            """"
            Noise filter (if rms < treshhold) = processing
            """
            if np.mean(rms) < activity_threshold:
                continue

            zcr = librosa.feature.zero_crossing_rate(chunk)[0]

            rms_expanded = expand_feature(rms, len(chunk))
            zcr_expanded = expand_feature(zcr, len(chunk))

            """
            Pitch
            """

            f0, voiced_flag, voiced_probs = librosa.pyin(
                chunk,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )

            f0 = np.nan_to_num(f0)
            pitch_expanded = expand_feature(f0, len(chunk))

            """
            Harmonic energy
            """

            harmonic, percussive = librosa.effects.hpss(chunk)

            harmonic_rms = librosa.feature.rms(y=harmonic)[0]
            harmonic_expanded = expand_feature(
                harmonic_rms,
                len(chunk)
            )

            """
            Frequency Modulation
            """

            pitch_delta = np.gradient(f0)

            pitch_delta_expanded = expand_feature(
                pitch_delta,
                len(chunk)
            )

            """
            Spectral Centroid
            """

            centroid = librosa.feature.spectral_centroid(
                y=chunk,
                sr=sr
            )[0]

            centroid_expanded = expand_feature(
                centroid,
                len(chunk)
            )

            """
            Spectral Bandwidth
            """

            bandwidth = librosa.feature.spectral_bandwidth(
                y=chunk,
                sr=sr
            )[0]

            bandwidth_expanded = expand_feature(
                bandwidth,
                len(chunk)
            )

            """
            Spectral rolloff
            """

            rolloff = librosa.feature.spectral_rolloff(
                y=chunk,
                sr=sr
            )[0]

            rolloff_expanded = expand_feature(
                rolloff,
                len(chunk)
            )

            """
            Spectral flatness
            """

            flatness = librosa.feature.spectral_flatness(
                y=chunk
            )[0]

            flatness_expanded = expand_feature(
                flatness,
                len(chunk)
            )

            """
            Syllable Structure
            """

            energy = librosa.feature.rms(y=chunk)[0]
            threshold = np.mean(energy)
            syllables = energy > threshold
            syllable_count = np.sum(
                np.diff(syllables.astype(int)) == 1
            )

            syllable_feature = np.full(
                len(chunk),
                syllable_count
            )

            """
            MFCC
            """

            mfcc = librosa.feature.mfcc(
                y=chunk,
                sr=sr,
                n_mfcc=13
            )

            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)

            delta_mfcc = delta_mfcc.T
            delta2_mfcc = delta2_mfcc.T

            # =========================
            # Stack channels
            # =========================

            rms_rehsaped = rms.reshape(-1, 1)
            zcr_rehsaped = zcr.reshape(-1, 1)
            f0_rehsaped = f0.reshape(-1, 1)
            pitch_delta_rehsaped = pitch_delta.reshape(-1, 1)
            centroid_rehsaped = centroid.reshape(-1, 1)
            bandwidth_rehsaped = bandwidth.reshape(-1, 1)
            rolloff_rehsaped = rolloff.reshape(-1, 1)
            flatness_rehsaped = flatness.reshape(-1, 1)

            mfcc = mfcc.T

            # print(rms.shape)
            # print(mfcc.shape)

            combined = np.concatenate([
                rms_rehsaped,
                zcr_rehsaped,
                f0_rehsaped,
                pitch_delta_rehsaped,
                centroid_rehsaped,
                bandwidth_rehsaped,
                rolloff_rehsaped,
                flatness_rehsaped,
                mfcc,
                delta_mfcc,
                delta2_mfcc
            ], axis=-1)

            # combinedOriginal = np.stack([
            #     normalize_feature(chunk),
            #
            #     rms_expanded,
            #     zcr_expanded,
            #     pitch_expanded,
            #     harmonic_expanded,
            #     pitch_delta_expanded,
            #     centroid_expanded,
            #     bandwidth_expanded,
            #     rolloff_expanded,
            #     flatness_expanded,
            #     # mfcc
            #     # mfcc,
            #     # syllable_feature
            # ], axis=-1)

            # print(combined.shape)
            processed_chunks.append(combined)

        return processed_chunks


    features = ["RMS", "ZCR", "Pitch", "Pitch Delta", "Centroid", "Bandwidth", "Rolloff", "Flatness", "mfcc",
                "delta_mfcc", "delta2_mfcc"]

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

    """
    Main Procession
    Data type (Sound, Spectrogram etc) is chosen after try: cycle starts
    """
    def process_dataset_split(dataset_path, test_size=0.2):
        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []
        test_file_ids = []

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
            print(f"Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")

            """check for simmilar files in several sets"""
            BirdsModels.checkIntersections(train_files, val_files, test_files)

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

                        test_file_ids.append(file)

                except Exception as e:
                    print(f"Error processing {file}: {e}")

        return (
            np.array(X_train), np.array(y_train),
            np.array(X_val), np.array(y_val),
            np.array(X_test), np.array(y_test),
            np.array(test_file_ids)
        )

    le = LabelEncoder()

    if os.path.isfile("D:/PythonProjects/MagRobotaData/BirdsData/X_train.npy"):
        X_train = np.load("D:/PythonProjects/MagRobotaData/BirdsData/X_train.npy")
        X_val = np.load("D:/PythonProjects/MagRobotaData/BirdsData/X_val.npy")
        X_test = np.load("D:/PythonProjects/MagRobotaData/BirdsData/X_test.npy")
        y_train = np.load("D:/PythonProjects/MagRobotaData/BirdsData/y_train.npy")
        y_val = np.load("D:/PythonProjects/MagRobotaData/BirdsData/y_val.npy")
        y_test = np.load("D:/PythonProjects/MagRobotaData/BirdsData/y_test.npy")
        test_file_ids = np.load("D:/PythonProjects/MagRobotaData/BirdsData/test_file_ids.npy")

        le = LabelEncoder()

        y_train = le.fit_transform(y_train)
        y_val = le.transform(y_val)
        y_test = le.transform(y_test)

        print("X_train shape:", X_train.shape)
        print("X_val shape:", X_val.shape)
        print("X_test shape:", X_test.shape)

        # print("Train:", Counter(y_train))
        # print("Val:", Counter(y_val))
        # print("Test:", Counter(y_test))

        # for cls in np.unique(y_train):
        #     idx = np.where(y_train == cls)[0]
        #
        #     print(f"\nClass {cls}")
        #
        #     mean_features = X_train[idx].mean(axis=(0, 1))
        #
        #     print(mean_features)
        # for i in range(X_train.shape[-1]):
        #     print(
        #         f"Feature {i}",
        #         np.min(X_train[:, :, i]),
        #         np.max(X_train[:, :, i]),
        #         np.std(X_train[:, :, i])
        #     )
        #
        # print(mean_features)
    else:
        X_train, y_train, X_val, y_val, X_test, y_test, test_file_ids = process_dataset_split(DATASET_PATH)
        print("X_train shape:", X_train.shape)
        print("X_val shape:", X_val.shape)
        print("X_test shape:", X_test.shape)

        # Encode labels
        le = LabelEncoder()
        # y_encoded = le.fit_transform(y)

        y_train = le.fit_transform(y_train)
        y_val = le.transform(y_val)
        y_test = le.transform(y_test)

        print(len(np.unique(y_train)))

        np.save("D:/PythonProjects/MagRobotaData/BirdsData/X_train.npy", X_train)
        np.save("D:/PythonProjects/MagRobotaData/BirdsData/X_val.npy", X_val)
        np.save("D:/PythonProjects/MagRobotaData/BirdsData/X_test.npy", X_test)
        np.save("D:/PythonProjects/MagRobotaData/BirdsData/y_train.npy", y_train)
        np.save("D:/PythonProjects/MagRobotaData/BirdsData/y_test.npy", y_test)
        np.save("D:/PythonProjects/MagRobotaData/BirdsData/y_val.npy", y_val)
        np.save("D:/PythonProjects/MagRobotaData/BirdsData/test_file_ids.npy", test_file_ids)

        # export_to_excel(X_train, y_train, le, "train_data.xlsx")
        # export_to_excel(X_val, y_val, le, "validation_data.xlsx")
        # export_to_excel(X_test, y_test, le, "test_data.xlsx")
        # export_full_chunks(X_val, y_val, le, "validation_data_full.xlsx")

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

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )

    class_weights = dict(
        zip(np.unique(y_train), weights)
    )

    print(class_weights)
    print(Counter(y_train))

    """
    Predictions
    """
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
        keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
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
    70% on 6 epochs - Good one
    - Amp
    - RMS
    - ZCR
    """

    modelCnn = BirdsModels.build_Cnn_model((X_train.shape[1], X_train.shape[2]), len(np.unique(y_train)))

    """
    70% 
    20 samples per file
    Long time, 10 > Epochs
    """
    modelInception = BirdsModels.build_inception_time((X_train.shape[1], X_train.shape[2]), len(np.unique(y_train)))


    """
    test
    """

    modelTest = Sequential([
        Conv1D(64, 7, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        MaxPooling1D(8),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(8),
        # keras.layers.Bidirectional(LSTM(64, return_sequences=False)),
        Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])

    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu', padding='same', input_shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        keras.layers.Dropout(0.4),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])

    modelConvLSTMSimple = Sequential([
        Conv1D(64, 3, padding='same', activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.BatchNormalization(),
        Conv1D(128, 3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Bidirectional(keras.layers.LSTM(64,return_sequences=False)),
        Dense(128, activation='relu'),
        keras.layers.Dropout(0.4),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])

    model5ConvLayers = Sequential([
        Conv1D(64, 5, padding='same', activation='relu',
               input_shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.BatchNormalization(),
        Conv1D(64, 5, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128, 3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        Conv1D(128, 3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(256, 3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling1D(),
        Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])


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
        epochs=50,
        batch_size=32,
        shuffle=True,
        callbacks=[early_stop],
        class_weight=class_weights
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = y_pred_probs.argmax(axis=1)

    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, y_pred))

    randomPred = show_random_predictions(X_test, y_test, model, le, n=20)

    """Prediction by chunks with file name, true and pred chunk"""
    BirdsModels.accuracyByFile(y_pred_probs, y_test, test_file_ids)
    """Prediction by file with file name, true and pred class, number of correct predictions of all number of chunks
    and TRUE of FALSE result"""
    BirdsModels.filePredictions(model, X_test, y_test, test_file_ids)

    if test_acc >= 0.6 and randomPred >= 5:
        modelName = model.config["type"] + str(f"_{test_acc:.5f}")
        model.save(f"D:/PythonProjects/MagRobotaData/BirdsData/Models/{modelName}/birds_model.keras")

        with open(f"D:/PythonProjects/MagRobotaData/BirdsData/Models/{modelName}/label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)
            config = {
            "sample_rate": SAMPLE_RATE,
            "chunk_size": CHUNK_SIZE,
            "channels": 3,
            "features": [
                "rms",
                "zcr",
                "pitch",
                "pitch_delta",
                "centroid",
                "bandwidth",
                "rolloff",
                "flatness",
                "mfcc",
                "delta_mfcc",
                "delta2_mfcc"
            ],
            "activity_detection": "40th percentile RMS"
        }

        with open(f"D:/PythonProjects/MagRobotaData/BirdsData/Models/{modelName}/config.json", "w") as f:
            json.dump(config, f)

        print('Model Saved!')








