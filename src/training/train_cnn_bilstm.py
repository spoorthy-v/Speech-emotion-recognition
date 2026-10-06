import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam

from src.datasets.dataloader import create_dataloaders
from src.models.cnn_bilstm import CNNBiLSTM


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(audio)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * audio.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            outputs = model(audio)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * audio.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    return (
        epoch_loss,
        epoch_accuracy,
        macro_f1,
        weighted_f1,
    )


def save_training_history(history, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
    ]

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)


def main():
    # Reproducibility
    set_seed(42)

    metadata_path = "data/metadata/ravdess_splits.csv"

    batch_size = 16
    learning_rate = 0.001
    num_epochs = 30

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    history_path = results_dir / "training_history_cnn_bilstm.csv"
    checkpoint_path = results_dir / "best_cnn_bilstm.pt"

    device = torch.device("cpu")

    print("Using device:", device)
    print("Model: CNN-BiLSTM")
    print("Random seed: 42")

    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_workers=0,
    )

    model = CNNBiLSTM(
        num_classes=8,
        hidden_size=128,
        num_layers=1,
        dropout=0.3,
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = []
    best_macro_f1 = -1.0

    for epoch in range(num_epochs):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        (
            val_loss,
            val_accuracy,
            val_macro_f1,
            val_weighted_f1,
        ) = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_weighted_f1,
        }

        history.append(epoch_results)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.4f} "
            f"Val Macro-F1: {val_macro_f1:.4f} "
            f"Val Weighted-F1: {val_weighted_f1:.4f}"
        )

        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = val_macro_f1

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_macro_f1,
                    "val_weighted_f1": val_weighted_f1,
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )

            print(
                f"  -> Best model saved "
                f"(Val Macro-F1: {val_macro_f1:.4f})"
            )

    save_training_history(
        history,
        history_path,
    )

    print("\nTraining completed.")
    print(f"Best validation Macro-F1: {best_macro_f1:.4f}")
    print(f"Best model: {checkpoint_path}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()