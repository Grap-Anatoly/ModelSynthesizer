import keras.layers

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


def build_model(input_shape, num_classes):
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