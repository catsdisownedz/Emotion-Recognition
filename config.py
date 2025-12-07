import os
from datetime import datetime

# ==========================================================
# EMOTION LABEL ORDER (CRITICAL FOR THE ENTIRE PROJECT)
# ==========================================================
# This order is used by:
# - Person A (preprocessing)
# - Person B (training + predictions)
# - Person C (GUI output)
# - Final JSON response
# DO NOT CHANGE THIS ORDER.

EMOTION_LABELS = [
    "neutral",     # 0
    "calm",        # 1
    "happy",       # 2
    "sad",         # 3
    "angry",       # 4
    "fearful",     # 5
    "disgust",     # 6
    "surprised"    # 7
]

NUM_CLASSES = len(EMOTION_LABELS)

# ==========================================================
# INPUT SHAPE FOR MODEL
# ==========================================================
INPUT_SHAPE = (128, 128, 1)   # Height, Width, Channels

# ==========================================================
# DATA DIRECTORIES
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_BASE_DIR = os.path.join(BASE_DIR, "models")

# ==========================================================
# FUNCTION: CREATE NEW TIMESTAMPED TRAIN RUN FOLDER
# ==========================================================
def create_run_folder():
    timestamp = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(MODEL_BASE_DIR, timestamp)

    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# ==========================================================
# FILE NAMES USED FOR SAVING MODEL + METRICS
# ==========================================================
BEST_MODEL_NAME = "best_model.h5"
METRICS_FILE = "metrics.json"
TRAINING_CURVES_FILE = "training_curves.png"
CONFUSION_MATRIX_FILE = "confusion_matrix.png"
CLASSIFICATION_REPORT_FILE = "classification_report.txt"

