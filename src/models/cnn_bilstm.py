import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        num_classes=8,
        hidden_size=128,
        num_layers=1,
        dropout=0.3,
    ):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Convert the frequency dimension into a fixed-size
        # representation while preserving the time dimension.
        self.frequency_pool = nn.AdaptiveAvgPool2d(
            (1, None)
        )

        # Temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else dropout,
        )

        self.dropout = nn.Dropout(dropout)

        # BiLSTM produces hidden_size * 2 features
        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes,
        )

    def forward(self, x):

        # x:
        # [batch, 1, mel_bins, time]
        x = self.cnn(x)

        # [batch, 128, 32, time]
        x = self.frequency_pool(x)

        # [batch, 128, 1, time]
        x = x.squeeze(2)

        # [batch, 128, time]
        x = x.transpose(1, 2)

        # [batch, time, 128]
        x, _ = self.lstm(x)

        # Global temporal pooling
        x = x.mean(dim=1)

        # [batch, hidden_size * 2]
        x = self.dropout(x)

        # [batch, num_classes]
        x = self.classifier(x)

        return x