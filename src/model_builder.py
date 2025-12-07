"""
Model builder for Speech Emotion Recognition (RAVDESS)
Architecture: CNN + BiLSTM + Dense with BatchNorm & Dropout
Input:  (128, 128, 1)  log-mel spectrogram
Output: (8,) softmax over emotions in this order:
    ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
"""

from tensorflow.keras import layers, models, optimizers


def build_emotion_model(input_shape=(128, 128, 1), num_classes=8) -> models.Model:
    """
    Builds a CNN + BiLSTM model for speech emotion recognition.

    Parameters
    ----------
    input_shape : tuple
        Shape of a single input example (H, W, C). We use (128, 128, 1).
    num_classes : int
        Number of emotion classes (8 for RAVDESS).

    Returns
    -------
    model : keras.Model
        Uncompiled Keras model.
    """

    inputs = layers.Input(shape=input_shape)

    # ---- CNN BLOCK 1 ----
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)     # 128x128 → 64x64
    x = layers.Dropout(0.25)(x)

    # ---- CNN BLOCK 2 ----
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)     # 64x64 → 32x32
    x = layers.Dropout(0.3)(x)

    # ---- CNN BLOCK 3 ----
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)     # 32x32 → 16x16
    x = layers.Dropout(0.35)(x)

    # At this point, with input (128,128,1):
    # x shape ≈ (batch, 16, 16, 128)

    # ---- RESHAPE FOR BiLSTM ----
    # Treat 16 "rows" as time steps, each with 16*128 features
    x = layers.Reshape((16, 16 * 128))(x)  # (batch, timesteps=16, features=2048)

    # ---- BiLSTM ----
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False)
    )(x)
    x = layers.Dropout(0.3)(x)

    # ---- DENSE CLASSIFIER HEAD ----
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_bilstm_emotion")

    return model
