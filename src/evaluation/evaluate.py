import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.datasets.dataloader import create_dataloaders
from src.models.cnn import CNNBaseline


EMOTION_LABELS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


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
    # Paths
    metadata_path = "data/metadata/ravdess_splits.csv"
    checkpoint_path = "results/best_cnn_baseline.pt"

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cpu")

    print("Using device:", device)

    # Data
    _, _, test_loader = create_dataloaders(
        metadata_path=metadata_path,
        batch_size=16,
        num_workers=0,
    )

    print("Test samples:", len(test_loader.dataset))

    # Model
    model = CNNBaseline(num_classes=8)
    model = model.to(device)

    # Load best checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Loaded checkpoint from epoch:",
        checkpoint["epoch"],
    )

    print(
        "Validation Macro-F1:",
        f"{checkpoint['val_macro_f1']:.4f}",
    )

    # Test evaluation
    labels, predictions = evaluate_model(
        model,
        test_loader,
        device,
    )

    # Overall metrics
    test_accuracy = accuracy_score(
        labels,
        predictions,
    )

    test_macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    test_weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\nTest Results")
    print("=" * 50)

    print(
        f"Accuracy:       {test_accuracy:.4f}"
    )

    print(
        f"Macro-F1:       {test_macro_f1:.4f}"
    )

    print(
        f"Weighted-F1:    {test_weighted_f1:.4f}"
    )

    # Classification report
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(EMOTION_LABELS))),
        target_names=EMOTION_LABELS,
        zero_division=0,
    )

    print("\nClassification Report")
    print("=" * 50)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(EMOTION_LABELS))),
    )

    print("\nConfusion Matrix")
    print("=" * 50)
    print(cm)

    # Save numerical results
    metrics = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_macro_f1": float(
            checkpoint["val_macro_f1"]
        ),
        "validation_weighted_f1": float(
            checkpoint["val_weighted_f1"]
        ),
        "validation_accuracy": float(
            checkpoint["val_accuracy"]
        ),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(
            test_weighted_f1
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(len(EMOTION_LABELS))),
            target_names=EMOTION_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = results_dir / "test_metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        f"\nSaved evaluation results to: {metrics_path}"
    )


if __name__ == "__main__":
    main()