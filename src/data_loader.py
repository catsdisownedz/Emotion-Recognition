import numpy as np
from sklearn.model_selection import train_test_split
import os

# Paths (can be overridden externally)
DATA_DIR = "notebooks/data"
X_FILE = "X.npy"
Y_FILE = "y.npy"

EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]


def load_data(data_dir=DATA_DIR):
    """Load X.npy and y.npy from disk."""
    X = np.load(os.path.join(data_dir, X_FILE))
    y = np.load(os.path.join(data_dir, Y_FILE))
    print("Loaded X:", X.shape)
    print("Loaded y:", y.shape)
    return X, y


def load_train_val_test(
    data_dir=DATA_DIR,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
    one_hot=True
):
    """Return proper train/val/test splits with matching shapes."""

    X, y = load_data(data_dir)

    # First split: train vs temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(test_size + val_size),
        stratify=y,
        random_state=random_state
    )

    # Second split: val vs test (equal split of temp)
    relative_val_size = val_size / (test_size + val_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - relative_val_size,  # test portion
        stratify=y_temp,
        random_state=random_state
    )

    # One-hot encoding
    if one_hot:
        num_classes = len(EMOTION_LABELS)

        y_train = np.eye(num_classes)[y_train]
        y_val   = np.eye(num_classes)[y_val]
        y_test  = np.eye(num_classes)[y_test]

    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, EMOTION_LABELS
