import librosa
import numpy as np
import torch


class MelSpectrogram:
    """Convert waveform into a log-Mel spectrogram."""

    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max,
        )

        return torch.tensor(
            mel_db,
            dtype=torch.float32,
        )