"""Create a reproducible binary classification split from MVTec bottle images."""

import random
import shutil
from pathlib import Path


RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_NAMES = ("train", "val", "test")
CLASS_NAMES = ("normal", "defective")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "bottle"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

SOURCE_DIRECTORIES = {
    "normal": (
        ("train_good", Path("train/good")),
        ("test_good", Path("test/good")),
    ),
    "defective": (
        ("broken_large", Path("test/broken_large")),
        ("broken_small", Path("test/broken_small")),
        ("contamination", Path("test/contamination")),
    ),
}


def collect_source_images() -> dict[str, list[tuple[Path, str]]]:
    """Collect product images and assign collision-safe destination names."""
    images_by_class: dict[str, list[tuple[Path, str]]] = {}

    for class_name, sources in SOURCE_DIRECTORIES.items():
        class_images: list[tuple[Path, str]] = []
        destination_names: set[str] = set()

        for filename_prefix, relative_directory in sources:
            source_directory = RAW_ROOT / relative_directory
            if not source_directory.is_dir():
                raise FileNotFoundError(f"Missing source directory: {source_directory}")

            for source_path in sorted(source_directory.iterdir()):
                if not source_path.is_file():
                    continue
                if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                destination_name = f"{filename_prefix}_{source_path.name}"
                if destination_name in destination_names:
                    raise RuntimeError(
                        f"Generated duplicate filename: {destination_name}"
                    )

                destination_names.add(destination_name)
                class_images.append((source_path, destination_name))

        images_by_class[class_name] = class_images

    return images_by_class


def split_images(
    images_by_class: dict[str, list[tuple[Path, str]]],
) -> dict[str, dict[str, list[tuple[Path, str]]]]:
    """Shuffle and split each class independently using the fixed seed."""
    assignments = {
        split_name: {class_name: [] for class_name in CLASS_NAMES}
        for split_name in SPLIT_NAMES
    }
    random_generator = random.Random(RANDOM_SEED)

    for class_name in CLASS_NAMES:
        class_images = images_by_class[class_name].copy()
        random_generator.shuffle(class_images)

        train_end = int(len(class_images) * TRAIN_RATIO)
        val_end = train_end + int(len(class_images) * VAL_RATIO)

        assignments["train"][class_name] = class_images[:train_end]
        assignments["val"][class_name] = class_images[train_end:val_end]
        assignments["test"][class_name] = class_images[val_end:]

    return assignments


def reset_processed_splits() -> None:
    """Safely clear only generated train, val, and test split contents."""
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    resolved_root = PROCESSED_ROOT.resolve()

    for split_name in SPLIT_NAMES:
        split_directory = PROCESSED_ROOT / split_name
        resolved_split = split_directory.resolve()
        if resolved_split.parent != resolved_root:
            raise RuntimeError(f"Unsafe processed split path: {resolved_split}")

        if split_directory.exists():
            for item in split_directory.iterdir():
                if item.is_symlink() or item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

        for class_name in CLASS_NAMES:
            (split_directory / class_name).mkdir(parents=True, exist_ok=True)


def copy_images(
    assignments: dict[str, dict[str, list[tuple[Path, str]]]],
) -> None:
    """Copy assigned images into the generated processed directories."""
    for split_name in SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            destination_directory = PROCESSED_ROOT / split_name / class_name
            for source_path, destination_name in assignments[split_name][class_name]:
                shutil.copy2(source_path, destination_directory / destination_name)


def validate_split(
    images_by_class: dict[str, list[tuple[Path, str]]],
    assignments: dict[str, dict[str, list[tuple[Path, str]]]],
) -> None:
    """Validate source assignment, class coverage, exclusions, and copied files."""
    expected_sources = {
        source_path.resolve()
        for class_images in images_by_class.values()
        for source_path, _ in class_images
    }
    assigned_sources = [
        source_path.resolve()
        for split_name in SPLIT_NAMES
        for class_name in CLASS_NAMES
        for source_path, _ in assignments[split_name][class_name]
    ]
    split_source_sets = {
        split_name: {
            source_path.resolve()
            for class_name in CLASS_NAMES
            for source_path, _ in assignments[split_name][class_name]
        }
        for split_name in SPLIT_NAMES
    }

    all_assigned_once = (
        len(assigned_sources) == len(expected_sources)
        and set(assigned_sources) == expected_sources
    )
    no_split_overlap = all(
        split_source_sets[first].isdisjoint(split_source_sets[second])
        for index, first in enumerate(SPLIT_NAMES)
        for second in SPLIT_NAMES[index + 1 :]
    )
    ground_truth_excluded = all(
        "ground_truth" not in source_path.relative_to(RAW_ROOT.resolve()).parts
        for source_path in assigned_sources
    )
    both_classes_present = all(
        assignments[split_name][class_name]
        for split_name in SPLIT_NAMES
        for class_name in CLASS_NAMES
    )
    copied_file_count = sum(
        1
        for split_name in SPLIT_NAMES
        for class_name in CLASS_NAMES
        for path in (PROCESSED_ROOT / split_name / class_name).iterdir()
        if path.is_file()
    )
    copied_count_matches = copied_file_count == len(expected_sources)

    checks = (
        ("All product images assigned exactly once", all_assigned_once),
        ("No image assigned to more than one split", no_split_overlap),
        ("Ground-truth masks excluded", ground_truth_excluded),
        ("Both classes present in every split", both_classes_present),
        ("Copied file count matches source count", copied_count_matches),
    )

    print("\nValidation checks:")
    for description, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {description}")

    if not all(passed for _, passed in checks):
        raise RuntimeError("Processed dataset validation failed.")


def print_summary(
    assignments: dict[str, dict[str, list[tuple[Path, str]]]],
) -> None:
    """Print class counts for each generated split."""
    total = 0
    print("Split summary:")
    for split_name in SPLIT_NAMES:
        normal_count = len(assignments[split_name]["normal"])
        defective_count = len(assignments[split_name]["defective"])
        total += normal_count + defective_count
        print(f"  {split_name}/normal: {normal_count}")
        print(f"  {split_name}/defective: {defective_count}")
    print(f"  Total images: {total}")


def main() -> None:
    images_by_class = collect_source_images()
    assignments = split_images(images_by_class)
    reset_processed_splits()
    copy_images(assignments)
    print_summary(assignments)
    validate_split(images_by_class, assignments)


if __name__ == "__main__":
    main()
