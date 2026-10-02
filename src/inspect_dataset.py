"""Inspect the structure and readability of an image dataset."""

import argparse
from pathlib import Path

from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SAMPLE_SIZE = 5
NORMAL_CATEGORIES = {"train/good", "test/good"}
DEFECTIVE_CATEGORIES = {
    "test/broken_large",
    "test/broken_small",
    "test/contamination",
}


def find_images(dataset_dir: Path) -> list[Path]:
    """Return supported image files found recursively in the dataset directory."""
    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def inspect_dataset(dataset_dir: Path) -> None:
    """Print image counts, sample metadata, and unreadable image paths."""
    image_paths = find_images(dataset_dir)
    samples: list[tuple[Path, tuple[int, int], str]] = []
    readable_product_images: list[Path] = []
    unreadable_images: list[tuple[Path, str]] = []
    ground_truth_count = 0

    for image_path in image_paths:
        relative_path = image_path.relative_to(dataset_dir)
        is_ground_truth = relative_path.parts[0] == "ground_truth"
        if is_ground_truth:
            ground_truth_count += 1

        try:
            with Image.open(image_path) as image:
                image.load()
                if not is_ground_truth:
                    readable_product_images.append(image_path)
                if not is_ground_truth and len(samples) < SAMPLE_SIZE:
                    samples.append((image_path, image.size, image.mode))
        except Exception as error:
            unreadable_images.append((image_path, str(error)))

    category_counts: dict[str, int] = {}
    for image_path in readable_product_images:
        category = image_path.parent.relative_to(dataset_dir).as_posix()
        category_counts[category] = category_counts.get(category, 0) + 1

    normal_count = sum(category_counts.get(path, 0) for path in NORMAL_CATEGORIES)
    defective_count = sum(
        category_counts.get(path, 0) for path in DEFECTIVE_CATEGORIES
    )

    print(f"Dataset directory: {dataset_dir}")
    print(f"Total usable product images: {len(readable_product_images)}")
    print(
        "Ground-truth mask images excluded from classification count: "
        f"{ground_truth_count}"
    )
    print("\nUsable product images by directory/category:")
    if category_counts:
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}")
    else:
        print("  No product image directories found.")

    # This is a count summary only; no files or labels are changed.
    print("\nBinary-class summary:")
    print(f"  Normal: {normal_count}")
    print(f"  Defective: {defective_count}")

    print("\nSample image details:")
    if samples:
        for image_path, (width, height), color_mode in samples:
            relative_path = image_path.relative_to(dataset_dir)
            print(
                f"  {relative_path}: {width}x{height} pixels, mode={color_mode}"
            )
    else:
        print("  No readable images available for sampling.")

    print(f"\nUnreadable or corrupted images: {len(unreadable_images)}")
    for image_path, error in unreadable_images:
        relative_path = image_path.relative_to(dataset_dir)
        print(f"  {relative_path}: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect image counts, metadata, and file readability."
    )
    parser.add_argument("dataset_dir", type=Path, help="Path to the dataset directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()

    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")

    inspect_dataset(dataset_dir)


if __name__ == "__main__":
    main()
