"""Reusable CPU inference for the trained defect classifier."""

import logging
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.model import create_model


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model_finetuned.pth"


class DefectPredictor:
    """Load the trained model once and reuse it for CPU predictions."""

    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH):
        self.device = torch.device("cpu")
        self.checkpoint_path = Path(checkpoint_path)

        logger.info("Loading model checkpoint from %s", self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.class_to_idx = checkpoint["class_to_idx"]
        self.idx_to_class = {
            class_index: class_name
            for class_name, class_index in self.class_to_idx.items()
        }
        self.image_size = tuple(checkpoint["image_size"])

        training_configuration = checkpoint.get("training_configuration", {})
        unfreeze_final_block = (
            training_configuration.get("unfrozen_feature_block") == "features[12]"
        )
        self.model = create_model(
            num_classes=len(self.class_to_idx),
            freeze_features=True,
            unfreeze_final_feature_block=unfreeze_final_block,
            weights=None,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        logger.info("Model loaded successfully on CPU")

    def predict(self, image: Image.Image) -> dict[str, str | float]:
        """Return the predicted class and its softmax confidence."""
        image_tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_index = probabilities.max(dim=1)

        class_name = self.idx_to_class[predicted_index.item()]
        return {
            "predicted_class": class_name.capitalize(),
            "confidence": confidence.item(),
        }
