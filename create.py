import soundfile as sf
import numpy as np

sr = 16000
duration = 2
audio = np.random.randn(sr * duration) * 0.01

sf.write("test.wav", audio, sr)
