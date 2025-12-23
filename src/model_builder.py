"""
Model builder for Speech Emotion Recognition (RAVDESS)
Architecture: CNN + BiLSTM + Dense with BatchNorm & Dropout

Input:  (128, 128, 2)
    Channel 0: log-mel spectrogram
    Channel 1: pitch (F0)

Output: (8,) softmax over emotions in this order:
    ["neutral", "calm", "happy", "sad",
     "angry", "fearful", "disgust", "surprised"]
"""

from tensorflow.keras import layers, models


def build_emotion_model(input_shape=(128, 128, 2), num_classes=8) -> models.Model:

    inputs = layers.Input(shape=input_shape)

    # ---- CNN BLOCK 1 ----
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # ---- CNN BLOCK 2 ----
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # ---- CNN BLOCK 3 ----
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.35)(x)

    # ---- RESHAPE FOR BiLSTM ----
    x = layers.Reshape((16, 16 * 128))(x)

    # ---- BiLSTM ----
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False)
    )(x)
    x = layers.Dropout(0.3)(x)

    # ---- CLASSIFIER ----
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs,
                         name="cnn_bilstm_emotion")

    return model
