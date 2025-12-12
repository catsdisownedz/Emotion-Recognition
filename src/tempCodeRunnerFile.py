"""
predict.py
Speech Emotion Recognition - Inference Module
Outputs JSON for GUI or API integration.

Automatically loads the newest trained model from:
    notebooks/models/run_YYYY-MM-DD_HH-MM-SS/
"""

import os
import json
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model


# ============================================================
# 1. Label Mapping
# ============================================================

EMOTION_LABELS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]


# ============================================================
# 2. Auto-detect latest model run folder
# ============================================================

def get_latest_run():
    """Return newest model run directory inside notebooks/models/."""
    import glob

    run_dirs = sorted(glob.glob("notebooks/models/run_*"), reverse=True)

    if not run_dirs:
        raise FileNotFoundError(
            "No run_* folder found in notebooks/models/. Train a model first."
        )

    return run_dirs[0]


def get_latest_model_paths():
    """Return (model_path, mean_std_path) for newest run."""
    latest = get_latest_run()
    model_path = os.path.join(latest, "best_model.h5")
    stats_path = os.path.join(latest, "mean_std.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"best_model.h5 not found in: {latest}")

    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"mean_std.json not found in: {latest}")

    return model_path, stats_path


# ============================================================
# 3. Load Model + Normalization Stats
# ============================================================

def load_ser_model(model_path, stats_path):
    """Load trained model and normalization stats."""
    model = load_model(model_path)

    with open(stats_path, "r") as f:
        stats = json.load(f)

    mean = float(stats["mean"])
    std = float(stats["std"])

    return model, mean, std


# ============================================================
# 4. Audio → Mel Spectrogram Preprocessing
# ============================================================

def preprocess_audio(file_path, mean, std,
                     sr=16000, n_mels=128, duration=2.0):
    """Convert WAV file → normalized (128,128,1) log-mel spectrogram."""

    # Load audio
    audio, _ = librosa.load(file_path, sr=sr)

    target_len = int(sr * duration)

    # Pad or trim to fixed length
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Min-max scale
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())

    # Replace fix_length with safe pad/trim
    if log_mel.shape[1] < 128:
        pad_width = 128 - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)))
    elif log_mel.shape[1] > 128:
        log_mel = log_mel[:, :128]

    # Normalize
    log_mel = (log_mel - mean) / std

    # Add channel dimension
    log_mel = np.expand_dims(log_mel, axis=-1)

    # Return (1,128,128,1)
    return np.expand_dims(log_mel, axis=0)


# ============================================================
# 5. Prediction → JSON Output
# ============================================================

def predict_emotion(model, mel_tensor):
    """Return structured JSON prediction output."""

    preds = model.predict(mel_tensor)[0]

    best_idx = int(np.argmax(preds))
    best_emotion = EMOTION_LABELS[best_idx]
    confidence = float(preds[best_idx])

    prob_dict = {
        EMOTION_LABELS[i]: float(preds[i])
        for i in range(len(EMOTION_LABELS))
    }

    return {
        "emotion": best_emotion,
        "confidence": confidence,
        "probabilities": prob_dict
    }


# ============================================================
# 6. Main Entry (GUI/API)
# ============================================================

def predict_from_file(audio_path, model_path=None, stats_path=None):
    """Main function called by GUI or terminal."""

    if model_path is None or stats_path is None:
        model_path, stats_path = get_latest_model_paths()

    model, mean, std = load_ser_model(model_path, stats_path)

    mel_input = preprocess_audio(audio_path, mean, std)

    return predict_emotion(model, mel_input)


# ============================================================
# 7. CLI (Command Line Usage)
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True,
                        help="Path to WAV audio file")

    args = parser.parse_args()

    result = predict_from_file(args.file)
    print(json.dumps(result, indent=4))
