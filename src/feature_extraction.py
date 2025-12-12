import numpy as np
import librosa

# ===============================
# Global preprocessing constants
# ===============================
SR = 16000
DURATION = 2.0          # seconds
N_MELS = 128
TARGET_FRAMES = 128


def wav_to_logmel(file_path: str) -> np.ndarray:
    """
    Convert WAV file to log-mel spectrogram (128, 128, 1)
    This function is used by BOTH training and prediction.
    """

    audio, _ = librosa.load(file_path, sr=SR)

    target_len = int(SR * DURATION)

    # Pad / trim audio
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_mels=N_MELS
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Per-sample min-max normalization
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

    # Enforce time dimension
    if log_mel.shape[1] < TARGET_FRAMES:
        pad = TARGET_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)))
    else:
        log_mel = log_mel[:, :TARGET_FRAMES]

    # Add channel dimension → (128,128,1)
    log_mel = np.expand_dims(log_mel, axis=-1)

    return log_mel


def normalize_logmel(log_mel: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Apply global dataset normalization.
    """
    return (log_mel - mean) / std
