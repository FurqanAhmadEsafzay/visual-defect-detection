"""Evaluate the saved classifier checkpoint on the untouched test split."""

from pathlib import Path

import torch

from dataset import create_test_dataloader
from model import create_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model_finetuned.pth"
BASELINE_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "baseline_evaluation_results.txt"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "final_evaluation_results.txt"
BATCH_SIZE = 16


def safe_divide(numerator: int, denominator: int) -> float:
    """Return zero when a metric denominator is zero."""
    return numerator / denominator if denominator else 0.0


def calculate_class_metrics(records: list[dict], class_index: int) -> dict[str, float]:
    """Calculate one-vs-rest precision, recall, and F1 for one class."""
    true_positives = sum(
        record["actual_index"] == class_index
        and record["predicted_index"] == class_index
        for record in records
    )
    false_positives = sum(
        record["actual_index"] != class_index
        and record["predicted_index"] == class_index
        for record in records
    )
    false_negatives = sum(
        record["actual_index"] == class_index
        and record["predicted_index"] != class_index
        for record in records
    )

    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, true_positives + false_negatives)
    f1_score = safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1_score}


def read_baseline_metrics() -> dict[str, float]:
    """Read baseline comparison values from the preserved evaluation report."""
    baseline_metrics = {}
    for line in BASELINE_OUTPUT_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("Accuracy:"):
            baseline_metrics["accuracy"] = float(line.split(":", 1)[1])
        elif line.strip().startswith("defective:"):
            values = line.split(":", 1)[1].split(",")
            parsed_values = {
                key.strip(): float(value)
                for key, value in (item.split("=") for item in values)
            }
            baseline_metrics["defective_recall"] = parsed_values["recall"]
            baseline_metrics["defective_f1"] = parsed_values["f1"]

    required_metrics = {"accuracy", "defective_recall", "defective_f1"}
    if set(baseline_metrics) != required_metrics:
        raise ValueError("Could not read all baseline metrics from the saved report.")
    return baseline_metrics


def evaluate(model, test_loader, idx_to_class: dict[int, str], device):
    """Run CPU inference and return one prediction record per test image."""
    model.eval()
    records = []
    sample_index = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            for actual_index, predicted_index, confidence in zip(
                labels.tolist(), predictions.tolist(), confidences.tolist()
            ):
                image_path = Path(test_loader.dataset.samples[sample_index][0])
                sample_index += 1
                records.append(
                    {
                        "path": image_path.relative_to(DATA_DIR / "test").as_posix(),
                        "actual_index": actual_index,
                        "predicted_index": predicted_index,
                        "actual_class": idx_to_class[actual_index],
                        "predicted_class": idx_to_class[predicted_index],
                        "confidence": confidence,
                        "correct": actual_index == predicted_index,
                    }
                )

    return records


def main() -> None:
    device = torch.device("cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    training_configuration = checkpoint.get("training_configuration", {})
    unfrozen_feature_block = training_configuration.get("unfrozen_feature_block")
    if unfrozen_feature_block != "features[12]":
        raise ValueError(
            "Unsupported fine-tuning configuration: "
            f"{unfrozen_feature_block}"
        )

    required_classes = {"defective", "normal"}
    if set(class_to_idx) != required_classes:
        raise ValueError(f"Unexpected checkpoint classes: {class_to_idx}")

    test_loader, test_class_to_idx = create_test_dataloader(
        DATA_DIR, batch_size=BATCH_SIZE
    )
    if test_class_to_idx != class_to_idx:
        raise ValueError(
            "Checkpoint and test dataset class mappings do not match: "
            f"{class_to_idx} != {test_class_to_idx}"
        )

    model = create_model(
        num_classes=len(class_to_idx),
        freeze_features=True,
        unfreeze_final_feature_block=True,
        weights=None,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    records = evaluate(model, test_loader, idx_to_class, device)
    defective_index = class_to_idx["defective"]
    normal_index = class_to_idx["normal"]

    true_positives = sum(
        record["actual_index"] == defective_index
        and record["predicted_index"] == defective_index
        for record in records
    )
    false_positives = sum(
        record["actual_index"] == normal_index
        and record["predicted_index"] == defective_index
        for record in records
    )
    false_negatives = sum(
        record["actual_index"] == defective_index
        and record["predicted_index"] == normal_index
        for record in records
    )
    true_negatives = sum(
        record["actual_index"] == normal_index
        and record["predicted_index"] == normal_index
        for record in records
    )

    accuracy = safe_divide(
        sum(record["correct"] for record in records), len(records)
    )
    metrics = {
        class_name: calculate_class_metrics(records, class_index)
        for class_name, class_index in class_to_idx.items()
    }
    macro_precision = sum(item["precision"] for item in metrics.values()) / len(
        metrics
    )
    macro_recall = sum(item["recall"] for item in metrics.values()) / len(metrics)
    macro_f1 = sum(item["f1"] for item in metrics.values()) / len(metrics)

    false_positive_records = [
        record
        for record in records
        if record["actual_index"] == normal_index
        and record["predicted_index"] == defective_index
    ]
    false_negative_records = [
        record
        for record in records
        if record["actual_index"] == defective_index
        and record["predicted_index"] == normal_index
    ]
    prediction_distribution = {
        class_name: sum(
            record["predicted_index"] == class_index for record in records
        )
        for class_name, class_index in class_to_idx.items()
    }
    baseline_metrics = read_baseline_metrics()

    summary_lines = [
        f"Checkpoint: {CHECKPOINT_PATH}",
        f"Checkpoint epoch: {checkpoint.get('epoch', 'not available')}",
        f"Device: {device}",
        f"Test samples: {len(records)}",
        f"Class mapping: {class_to_idx}",
        f"Accuracy: {accuracy:.6f}",
        f"Macro Precision: {macro_precision:.6f}",
        f"Macro Recall: {macro_recall:.6f}",
        f"Macro F1: {macro_f1:.6f}",
        "",
        "Per-class metrics:",
        *[
            (
                f"  {class_name}: precision={values['precision']:.6f}, "
                f"recall={values['recall']:.6f}, f1={values['f1']:.6f}"
            )
            for class_name, values in sorted(
                metrics.items(), key=lambda item: class_to_idx[item[0]]
            )
        ],
        "",
        "Confusion matrix (rows=actual, columns=predicted):",
        "                 Predicted",
        "                 Defective   Normal",
        f"Actual Defective {true_positives:10d} {false_negatives:8d}",
        f"Actual Normal    {false_positives:10d} {true_negatives:8d}",
        "",
        "Prediction distribution:",
        f"  Predicted Defective: {prediction_distribution['defective']}",
        f"  Predicted Normal: {prediction_distribution['normal']}",
        "",
        "Defective as positive class:",
        f"  TP: {true_positives}",
        f"  FP: {false_positives}",
        f"  FN: {false_negatives}",
        f"  TN: {true_negatives}",
        "  False Positive: Normal product predicted as Defective.",
        "  False Negative: Defective product predicted as Normal.",
        "",
        "Baseline vs fine-tuned:",
        "  Baseline:",
        f"    Accuracy: {baseline_metrics['accuracy']:.6f}",
        f"    Defective Recall: {baseline_metrics['defective_recall']:.6f}",
        f"    Defective F1: {baseline_metrics['defective_f1']:.6f}",
        "  Fine-tuned:",
        f"    Accuracy: {accuracy:.6f}",
        f"    Defective Recall: {metrics['defective']['recall']:.6f}",
        f"    Defective F1: {metrics['defective']['f1']:.6f}",
    ]

    error_lines = ["", "FALSE POSITIVES"]
    if false_positive_records:
        error_lines.extend(
            f"{record['path']} | actual={record['actual_class']} | "
            f"predicted={record['predicted_class']} | "
            f"confidence={record['confidence']:.6f}"
            for record in false_positive_records
        )
    else:
        error_lines.append("None")

    error_lines.extend(["", "FALSE NEGATIVES"])
    if false_negative_records:
        error_lines.extend(
            f"{record['path']} | actual={record['actual_class']} | "
            f"predicted={record['predicted_class']} | "
            f"confidence={record['confidence']:.6f}"
            for record in false_negative_records
        )
    else:
        error_lines.append("None")

    prediction_lines = ["", "PER-IMAGE PREDICTIONS"]
    prediction_lines.extend(
        f"{record['path']} | actual={record['actual_class']} | "
        f"predicted={record['predicted_class']} | "
        f"confidence={record['confidence']:.6f} | "
        f"correct={record['correct']}"
        for record in records
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "\n".join(summary_lines + error_lines + prediction_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(summary_lines + error_lines))
    print(f"\nEvaluation results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
