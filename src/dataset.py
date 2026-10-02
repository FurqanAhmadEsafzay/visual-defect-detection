"""Create PyTorch DataLoaders for the processed defect dataset."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_BATCH_SIZE = 16
NUM_WORKERS = 0


def create_transforms():
    """Create augmented training and deterministic evaluation transforms."""
    train_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def create_test_dataloader(
    data_dir: str | Path, batch_size: int = DEFAULT_BATCH_SIZE
):
    """Create only the deterministic test DataLoader and its class mapping."""
    data_dir = Path(data_dir)
    _, evaluation_transform = create_transforms()
    test_dataset = datasets.ImageFolder(data_dir / "test", evaluation_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    return test_loader, test_dataset.class_to_idx


def create_dataloaders(data_dir: str | Path, batch_size: int = DEFAULT_BATCH_SIZE):
    """Create train, validation, and test loaders plus the discovered classes."""
    data_dir = Path(data_dir)
    train_transform, evaluation_transform = create_transforms()

    train_dataset = datasets.ImageFolder(data_dir / "train", train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", evaluation_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", evaluation_transform)

    class_to_idx = train_dataset.class_to_idx
    if val_dataset.class_to_idx != class_to_idx or test_dataset.class_to_idx != class_to_idx:
        raise ValueError("Class mappings are inconsistent across dataset splits.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader, class_to_idx


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "processed"

    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        data_dir
    )
    print(f"Class mapping: {class_to_idx}")
    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Test images: {len(test_loader.dataset)}")

    images, labels = next(iter(train_loader))
    print(f"Training image batch shape: {images.shape}")
    print(f"Training label batch shape: {labels.shape}")
    print(f"Unique labels in batch: {labels.unique().tolist()}")


if __name__ == "__main__":
    main()
