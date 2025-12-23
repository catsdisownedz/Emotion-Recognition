import numpy as np
import librosa

SR = 16000
DURATION = 2.0
N_MELS = 128
TARGET_FRAMES = 128

N_FFT = 1024
HOP_LENGTH = 256


def _fix_length_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[0] < target_len:
        return np.pad(x, (0, target_len - x.shape[0]))
    return x[:target_len]


def _fix_frames_2d(x: np.ndarray, target_frames: int) -> np.ndarray:
    # x shape: (features, frames)
    frames = x.shape[1]
    if frames < target_frames:
        return np.pad(x, ((0, 0), (0, target_frames - frames)))
    return x[:, :target_frames]


def wav_to_features(file_path: str) -> np.ndarray:
    """
    Deterministic features for SER.
    Output: (128, 128, 2) where:
      channel 0 = log-mel
      channel 1 = pitch track aligned to mel frames
    """
    audio, _ = librosa.load(file_path, sr=SR)

    # 1) Trim silence (deterministic)
    audio, _ = librosa.effects.trim(audio, top_db=25)

    # 2) Pad/trim fixed length
    target_len = int(SR * DURATION)
    audio = _fix_length_1d(audio, target_len)

    # 3) Log-mel
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=1.0)
    log_mel = _fix_frames_2d(log_mel, TARGET_FRAMES)  # (128,128)

    # 4) Pitch (F0) using YIN (robust + simple)
    # Compute pitch per analysis frame similar to hop_length
    f0 = librosa.yin(
        audio,
        fmin=50,
        fmax=500,
        sr=SR,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )  # shape ~ (frames,)

    # 5) Fix length to 128 frames
    f0 = _fix_length_1d(f0, TARGET_FRAMES)  # (128,)

    # 6) Stabilize pitch: replace NaN/inf, then log-scale
    f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
    f0 = np.log1p(f0)  # compress dynamic range

    # 7) Expand pitch into a 2D map aligned with mel
    pitch_map = np.tile(f0[None, :], (N_MELS, 1))  # (128,128)

    # 8) Stack channels
    feat = np.stack([log_mel, pitch_map], axis=-1).astype(np.float32)  # (128,128,2)
    return feat
