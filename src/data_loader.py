from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np

EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]

def load_train_val_test(X_path, y_path, test_size=0.15, val_size=0.15):
    X = np.load(X_path)
    y = np.load(y_path)

    y = to_categorical(y, num_classes=8)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size + val_size,
        stratify=np.argmax(y, axis=1),
        random_state=42
    )

    relative_test_size = test_size / (test_size + val_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=np.argmax(y_temp, axis=1),
        random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, EMOTION_LABELS
