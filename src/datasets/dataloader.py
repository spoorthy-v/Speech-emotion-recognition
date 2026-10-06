import torch
from torch.utils.data import DataLoader

from src.datasets.ser_dataset import RAVDESSDataset
from src.features.mel_spectrogram import MelSpectrogram


def create_dataloaders(
    metadata_path,
    batch_size=16,
    num_workers=0,
):
    feature_extractor = MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
    )

    train_dataset = RAVDESSDataset(
        metadata_path=metadata_path,
        split="train",
        sample_rate=16000,
        duration=3.0,
        feature_extractor=feature_extractor,
    )

    val_dataset = RAVDESSDataset(
        metadata_path=metadata_path,
        split="validation",
        sample_rate=16000,
        duration=3.0,
        feature_extractor=feature_extractor,
    )

    test_dataset = RAVDESSDataset(
        metadata_path=metadata_path,
        split="test",
        sample_rate=16000,
        duration=3.0,
        feature_extractor=feature_extractor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader