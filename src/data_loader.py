import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Global constants
NUM_CLASSES = 8
DATA_DIR = "data"    # folder where X.npy and y.npy are stored


def load_data():
    """
    Loads preprocessed arrays from .npy files.
    X.npy must be shape (N,128,128,1)
    y.npy must be shape (N,)
    """
    X = np.load(f"{DATA_DIR}/X.npy")
    y = np.load(f"{DATA_DIR}/y.npy")

    print("Loaded X:", X.shape)
    print("Loaded y:", y.shape)

    return X, y


def load_train_val_test(test_size=0.15, val_size=0.15):
    """
    Loads X and y from .npy files,
    one-hot encodes labels,
    and returns train/val/test splits.
    """

    # Step 1: Load raw arrays
    X, y = load_data()

    # Step 2: One-hot encode labels
    y_onehot = to_categorical(y, num_classes=NUM_CLASSES)

    # Step 3: Split into train + temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_onehot,
        test_size=(test_size + val_size),
        shuffle=True,
        random_state=42
    )

    # Step 4: Split temp into val + test
    relative_val = val_size / (test_size + val_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - relative_val,
        shuffle=True,
        random_state=42
    )

    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test
