import torch
import torch.nn as nn
from torch.optim import Adam

from src.datasets.dataloader import create_dataloaders
from src.models.cnn import CNNBaseline


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

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def main():
    # Configuration
    metadata_path = "data/metadata/ravdess_splits.csv"

    batch_size = 16
    learning_rate = 0.001
    num_epochs = 30

    # CPU for Intel Mac
    device = torch.device("cpu")

    print("Using device:", device)

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_workers=0,
    )

    # Model
    model = CNNBaseline(num_classes=8)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # Training
    for epoch in range(num_epochs):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()