"""Train the MobileNetV3 Small classifier on CPU."""

import random
import time
from math import isclose
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from dataset import IMAGE_SIZE, create_dataloaders
from model import MODEL_NAME, create_model


RANDOM_SEED = 42
BATCH_SIZE = 16
CLASSIFIER_LEARNING_RATE = 0.0005
FEATURE_BLOCK_LEARNING_RATE = 0.0001
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model_finetuned.pth"


def calculate_class_weights(train_dataset, class_to_idx: dict[str, int]) -> torch.Tensor:
    """Calculate inverse-frequency weights in ImageFolder class-index order."""
    class_counts = Counter(train_dataset.targets)
    total_samples = len(train_dataset)
    num_classes = len(class_to_idx)
    weights = torch.empty(num_classes, dtype=torch.float32)

    for class_name, class_index in class_to_idx.items():
        class_count = class_counts[class_index]
        if class_count == 0:
            raise ValueError(f"Training class has no images: {class_name}")
        weights[class_index] = total_samples / (num_classes * class_count)

    return weights


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """Train the classifier for one epoch and return loss and accuracy."""
    model.train()
    for frozen_block in model.features[:-1]:
        frozen_block.eval()
    running_loss = 0.0
    correct_predictions = 0
    sample_count = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
        sample_count += batch_size

    return running_loss / sample_count, correct_predictions / sample_count


def validate(model, data_loader, criterion, device, defective_index: int):
    """Calculate validation loss, accuracy, and Defective class metrics."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    sample_count = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            predictions = logits.argmax(dim=1)
            running_loss += loss.item() * batch_size
            correct_predictions += (predictions == labels).sum().item()
            sample_count += batch_size
            true_positives += (
                (predictions == defective_index) & (labels == defective_index)
            ).sum().item()
            false_positives += (
                (predictions == defective_index) & (labels != defective_index)
            ).sum().item()
            false_negatives += (
                (predictions != defective_index) & (labels == defective_index)
            ).sum().item()

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = (
        true_positives / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return (
        running_loss / sample_count,
        correct_predictions / sample_count,
        precision,
        recall,
        f1_score,
    )


def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cpu")

    train_loader, val_loader, _, class_to_idx = create_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE
    )
    class_weights = calculate_class_weights(train_loader.dataset, class_to_idx).to(
        device
    )

    defective_index = class_to_idx["defective"]
    model = create_model(
        num_classes=len(class_to_idx),
        freeze_features=True,
        unfreeze_final_feature_block=True,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    classifier_parameters = list(model.classifier.parameters())
    feature_block_parameters = list(model.features[-1].parameters())
    optimizer = Adam(
        [
            {
                "params": classifier_parameters,
                "lr": CLASSIFIER_LEARNING_RATE,
            },
            {
                "params": feature_block_parameters,
                "lr": FEATURE_BLOCK_LEARNING_RATE,
            },
        ]
    )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen_parameters = total_parameters - trainable_parameters

    print(f"Device: {device}")
    print(f"Class mapping: {class_to_idx}")
    print("Unfrozen feature block: features[12]")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Frozen parameters: {frozen_parameters:,}")
    print("Optimizer parameter groups:")
    print(f"  classifier: lr={CLASSIFIER_LEARNING_RATE}")
    print(f"  features[12]: lr={FEATURE_BLOCK_LEARNING_RATE}")
    print("Class weights:")
    for class_name, class_index in sorted(
        class_to_idx.items(), key=lambda item: item[1]
    ):
        class_count = train_loader.dataset.targets.count(class_index)
        print(
            f"  {class_name} (index {class_index}, count {class_count}): "
            f"{class_weights[class_index].item():.6f}"
        )

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_val_defective_precision = 0.0
    best_val_defective_recall = 0.0
    best_val_defective_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_completed = 0
    training_start = time.perf_counter()

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        (
            val_loss,
            val_accuracy,
            val_defective_precision,
            val_defective_recall,
            val_defective_f1,
        ) = validate(model, val_loader, criterion, device, defective_index)
        epochs_completed = epoch

        print(f"\nEpoch {epoch}/{MAX_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        print(f"Val Defective Precision: {val_defective_precision:.4f}")
        print(f"Val Defective Recall: {val_defective_recall:.4f}")
        print(f"Val Defective F1: {val_defective_f1:.4f}")

        f1_improved = val_defective_f1 > best_val_defective_f1
        f1_tied = isclose(val_defective_f1, best_val_defective_f1)
        checkpoint_improved = f1_improved or (f1_tied and val_loss < best_val_loss)

        if checkpoint_improved:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_val_defective_precision = val_defective_precision
            best_val_defective_recall = val_defective_recall
            best_val_defective_f1 = val_defective_f1
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "model_name": MODEL_NAME,
                    "image_size": IMAGE_SIZE,
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                    "best_val_defective_precision": best_val_defective_precision,
                    "best_val_defective_recall": best_val_defective_recall,
                    "best_val_defective_f1": best_val_defective_f1,
                    "epoch": epoch,
                    "training_configuration": {
                        "batch_size": BATCH_SIZE,
                        "maximum_epochs": MAX_EPOCHS,
                        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                        "random_seed": RANDOM_SEED,
                        "classifier_learning_rate": CLASSIFIER_LEARNING_RATE,
                        "feature_block_learning_rate": FEATURE_BLOCK_LEARNING_RATE,
                        "unfrozen_feature_block": "features[12]",
                    },
                },
                CHECKPOINT_PATH,
            )
            print("Best checkpoint updated.")

        if f1_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(
                "Validation Defective F1 did not improve "
                f"({epochs_without_improvement}/{EARLY_STOPPING_PATIENCE})."
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    elapsed_time = time.perf_counter() - training_start
    print("\nTraining complete")
    print(f"Epochs completed: {epochs_completed}")
    print(f"Best checkpoint epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best validation accuracy: {best_val_accuracy:.6f}")
    print(f"Best validation Defective precision: {best_val_defective_precision:.6f}")
    print(f"Best validation Defective recall: {best_val_defective_recall:.6f}")
    print(f"Best validation Defective F1: {best_val_defective_f1:.6f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Total training time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
