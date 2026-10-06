import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.datasets.dataloader import create_dataloaders
from src.models.cnn_bilstm import CNNBiLSTM


def evaluate_model(model, loader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            outputs = model(audio)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    return all_labels, all_predictions


def main():

    metadata_path = "data/metadata/ravdess_splits.csv"
    checkpoint_path = "results/best_cnn_bilstm.pt"

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "test_metrics_cnn_bilstm.json"

    device = torch.device("cpu")

    print("Using device:", device)

    # Create dataloaders using the same preprocessing
    # and speaker-independent split as the CNN baseline.
    _, _, test_loader = create_dataloaders(
        metadata_path=metadata_path,
        batch_size=16,
        num_workers=0,
    )

    # Create model
    model = CNNBiLSTM(
        num_classes=8,
        hidden_size=128,
        num_layers=1,
        dropout=0.3,
    )

    model = model.to(device)

    # Load best validation checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation Macro-F1: "
        f"{checkpoint['val_macro_f1']:.4f}"
    )

    # Evaluate on test set
    labels, predictions = evaluate_model(
        model,
        test_loader,
        device,
    )

    # Calculate metrics
    accuracy = accuracy_score(
        labels,
        predictions,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    # Emotion names
    emotion_names = [
        "neutral",
        "calm",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgust",
        "surprised",
    ]

    # Classification report
    report = classification_report(
        labels,
        predictions,
        labels=list(range(8)),
        target_names=emotion_names,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(
        labels,
        predictions,
        labels=list(range(8)),
    )

    # Print results
    print("\nTest Results")
    print("=" * 50)

    print(
        f"Accuracy:       {accuracy:.4f}"
    )

    print(
        f"Macro-F1:       {macro_f1:.4f}"
    )

    print(
        f"Weighted-F1:    {weighted_f1:.4f}"
    )

    print("\nClassification Report")
    print("=" * 50)
    print(report)

    print("Confusion Matrix")
    print("=" * 50)
    print(cm)

    # Save metrics
    results = {
        "model": "CNN-BiLSTM",
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "validation_macro_f1": float(
            checkpoint["val_macro_f1"]
        ),
        "validation_weighted_f1": float(
            checkpoint["val_weighted_f1"]
        ),
        "validation_accuracy": float(
            checkpoint["val_accuracy"]
        ),
        "test_accuracy": float(
            accuracy
        ),
        "test_macro_f1": float(
            macro_f1
        ),
        "test_weighted_f1": float(
            weighted_f1
        ),
        "confusion_matrix": cm.tolist(),
    }

    with open(
        metrics_path,
        "w",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print(
        f"\nSaved evaluation results to: "
        f"{metrics_path}"
    )


if __name__ == "__main__":
    main()