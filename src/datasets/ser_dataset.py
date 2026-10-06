from pathlib import Path

import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset


class RAVDESSDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        split,
        sample_rate=16000,
        duration=3.0,
        feature_extractor=None,
    ):
        self.df = pd.read_csv(metadata_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        audio_path = Path(row["file_path"])

        audio, _ = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=True,
        )

        # Fixed-length audio
        if len(audio) < self.num_samples:
            audio = torch.nn.functional.pad(
                torch.tensor(audio),
                (0, self.num_samples - len(audio)),
            ).numpy()
        else:
            audio = audio[:self.num_samples]

        audio = torch.tensor(audio, dtype=torch.float32)

        label = int(row["emotion_id"]) - 1

        # Apply feature extraction if provided
        if self.feature_extractor is not None:
            audio = self.feature_extractor(audio)

            # Add channel dimension for CNN
            audio = audio.unsqueeze(0)

        return {
            "audio": audio,
            "label": label,
            "emotion": row["emotion"],
            "actor_id": int(row["actor_id"]),
        }