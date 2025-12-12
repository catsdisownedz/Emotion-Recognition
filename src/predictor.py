# src/predictor.py
import os
import json
import glob
from typing import Dict, Any, Optional

import numpy as np
from tensorflow.keras.models import load_model

from src.feature_extraction import wav_to_logmel, normalize_logmel

EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]


def get_latest_run(models_root: str = "notebooks/models") -> str:
    run_dirs = sorted(
        glob.glob(os.path.join(models_root, "run_*")),
        reverse=True
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run_* folder found in {models_root}. Train a model first.")
    return run_dirs[0]


class EmotionPredictor:
    """
    Loads the latest trained model run (best_model.h5 + mean_std.json) once,
    then provides predict(audio_path) -> JSON dict for the GUI.
    """

    def __init__(self, run_dir: Optional[str] = None, models_root: str = "notebooks/models"):
        self.run_dir = run_dir or get_latest_run(models_root=models_root)

        self.model_path = os.path.join(self.run_dir, "best_model.h5")
        self.stats_path = os.path.join(self.run_dir, "mean_std.json")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not os.path.exists(self.stats_path):
            raise FileNotFoundError(f"mean_std.json not found: {self.stats_path}")

        self.model = load_model(self.model_path)

        with open(self.stats_path, "r") as f:
            stats = json.load(f)
        self.mean = float(stats["mean"])
        self.std = float(stats["std"])

    def predict(self, audio_path: str) -> Dict[str, Any]:
        mel = wav_to_logmel(audio_path)                 # (128,128,1)
        mel = normalize_logmel(mel, self.mean, self.std)
        x = np.expand_dims(mel, axis=0)                 # (1,128,128,1)

        probs = self.model.predict(x, verbose=0)[0]     # (8,)
        idx = int(np.argmax(probs))

        return {
            "emotion": EMOTION_LABELS[idx],
            "confidence": float(probs[idx]),
            "probabilities": {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}
        }
