import keras.layers
import numpy as np
from scipy.stats import skew, kurtosis
import tensorflow as tf
from collections import defaultdict

def extract_global_features(X):
    # X shape (n_samples, 200, 5)
    feats = []
    for sample in X:
        sample_feats = []
        for ch in range(sample.shape[1]):
            col = sample[:, ch]
            sample_feats.extend([
                np.mean(col), np.std(col),
                np.min(col), np.percentile(col, 25),
                np.median(col), np.percentile(col, 75),
                np.max(col), skew(col), kurtosis(col)
            ])
        feats.append(sample_feats)
    return np.array(feats)

def residual_block(x, filters, kernel_size, stride=1):
    """Basic residual block with two Conv1D layers and a skip connection."""
    shortcut = x
    # First conv
    x = keras.layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    # Second conv (always stride 1)
    x = keras.layers.Conv1D(filters, kernel_size, strides=1, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    # Adjust shortcut dimensions if needed
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = keras.layers.BatchNormalization()(shortcut)

    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.ReLU()(x)
    return x


def build_Cnn_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)  # (6615, 3)

    # Initial stem: reduce temporal length quickly
    x = keras.layers.Conv1D(32, kernel_size=7, strides=4, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    # Now shape ~ (1654, 32)

    # Residual blocks with progressive downsampling
    x = residual_block(x, filters=64, kernel_size=5, stride=2)  # (827, 64)
    x = residual_block(x, filters=128, kernel_size=5, stride=2)  # (414, 128)
    x = residual_block(x, filters=256, kernel_size=3, stride=2)  # (207, 256)
    x = residual_block(x, filters=256, kernel_size=3, stride=1)  # (207, 256)

    # Global pooling removes the temporal dimension
    x = keras.layers.GlobalAveragePooling1D()(x)

    # Classifier head with heavy dropout
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model


def build_cnn_lstm(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv1D(32, kernel_size=5, strides=2, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1D(64, kernel_size=5, strides=4, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)  # shape: (?, 827, 128)

    x = keras.layers.Conv1D(128, kernel_size=3, strides=4, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)  # shape: (?, 207, 256)

    x = keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)  # (?, 207, 128)

    x = keras.layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)(x)

    # Classifier
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model


def inception_module(x, filters=32):
    # Four parallel branches
    b1 = keras.layers.Conv1D(filters, 1, padding='same', activation='relu')(x)

    b2 = keras.layers.Conv1D(filters, 1, padding='same', activation='relu')(x)
    b2 = keras.layers.Conv1D(filters, 3, padding='same', activation='relu')(b2)

    b3 = keras.layers.Conv1D(filters, 1, padding='same', activation='relu')(x)
    b3 = keras.layers.Conv1D(filters, 5, padding='same', activation='relu')(b3)

    b4 = keras.layers.MaxPooling1D(3, strides=1, padding='same')(x)
    b4 = keras.layers.Conv1D(filters, 1, padding='same', activation='relu')(b4)

    return keras.layers.Concatenate()([b1, b2, b3, b4])

"""
Pure convolutional
70% 
20 samples per file
Long time, 10 > Epochs
"""
def build_inception_time(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for filters in [32, 64, 128]:
        x = inception_module(x, filters)
        x = keras.layers.MaxPooling1D(2)(x)  # halve the time dimension
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)


def transformer_encoder(x, num_heads=4, ff_dim=64, dropout=0.3):
    # Multi-head self-attention
    attn = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1], dropout=dropout)(x, x)
    attn = keras.layers.Add()([x, attn])
    attn = keras.layers.LayerNormalization()(attn)

    # Feed-forward
    ffn = keras.layers.Dense(ff_dim, activation='relu')(attn)
    ffn = keras.layers.Dense(x.shape[-1])(ffn)
    ffn = keras.layers.Add()([attn, ffn])
    ffn = keras.layers.LayerNormalization()(ffn)
    return ffn

"""
Random forest
"""

from sklearn.ensemble import RandomForestClassifier

def rfModel(X_train_rf, X_test_rf, y_train, y_test):
    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=15,
        class_weight='balanced',
        random_state=42
    )

    rf.fit(X_train_rf, y_train)

    score = rf.score(X_test_rf, y_test)
    print(score)

    return score



"""
Test values
"""

def permutationImportance(X_test, y_test, model, test_acc):
    # print(X_test)
    importanceSum = 0

    for i in range(47):
        X_perm = X_test.copy()

        np.random.shuffle(
            X_perm[:, :, i]
        )

        acc = model.evaluate(
            X_perm,
            y_test,
            verbose=0
        )[1]

        importanceSum += test_acc - acc

        print(f"{i} importance:", test_acc - acc)
    print(importanceSum)

def checkIntersections(train_files, val_files, test_files):
    common_files = list(set(train_files).intersection(val_files, test_files))

    print("Files in all 3 lists:", common_files)

    print(
        "Train-Val:",
        len(set(train_files) & set(val_files))
    )

    print(
        "Train-Test:",
        len(set(train_files) & set(test_files))
    )

    print(
        "Val-Test:",
        len(set(val_files) & set(test_files))
    )

def accuracyByFile(y_probs, y_test, test_file_ids):

    #print(test_file_ids)

    file_probs = defaultdict(list)
    file_labels = {}

    for probs, file_id, true_label in zip(
            y_probs,
            test_file_ids,
            y_test):

        file_probs[file_id].append(probs)
        file_labels[file_id] = true_label

    correct = 0

    for file_id in file_probs:

        avg_probs = np.mean(
            file_probs[file_id],
            axis=0
        )

        pred = np.argmax(avg_probs)

        confidence = np.max(avg_probs)

        true = file_labels[file_id]

        print()
        print("File:", file_id)
        print("True:", true)
        print("Pred:", pred)
        print("Confidence:", confidence)

        if pred == true:
            correct += 1

def filePredictions(model, X_test, y_test, test_file_ids):
    y_probs = model.predict(X_test)
    y_pred_chunks = np.argmax(y_probs, axis=1)

    # Group information by file
    file_chunk_preds = defaultdict(list)
    file_chunk_true = defaultdict(list)
    file_chunk_probs = defaultdict(list)

    for pred, true, probs, file_id in zip(
            y_pred_chunks,
            y_test,
            y_probs,
            test_file_ids):
        file_chunk_preds[file_id].append(pred)
        file_chunk_true[file_id].append(true)
        file_chunk_probs[file_id].append(probs)

    # Analyze each file
    for file_id in file_chunk_preds:

        chunk_preds = np.array(file_chunk_preds[file_id])
        chunk_true = np.array(file_chunk_true[file_id])

        true_label = chunk_true[0]

        # chunk accuracy
        correct_chunks = np.sum(chunk_preds == true_label)
        total_chunks = len(chunk_preds)

        # file prediction from averaged probabilities
        avg_probs = np.mean(
            file_chunk_probs[file_id],
            axis=0
        )

        file_pred = np.argmax(avg_probs)

        print()
        print(f"File: {file_id}")
        print(f"True class: {true_label}")
        print(f"File prediction: {file_pred}")
        print(f"Correct chunks: {correct_chunks}/{total_chunks}")

        if file_pred == true_label:
            print("Result: TRUE")
        else:
            print("Result: FALSE")

