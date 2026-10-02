"""FastAPI application for visual defect detection."""

import logging
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from src.inference import DefectPredictor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp"}

logger.info("Initializing Visual Defect Detection model")
predictor = DefectPredictor()
logger.info("Application initialization complete")

app = FastAPI(title="Visual Defect Detection API")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Visual Defect Detection API is running"}


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "healthy", "model_loaded": predictor is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, str | float]:
    logger.info(
        "Prediction request: filename=%s, content_type=%s",
        file.filename,
        file.content_type,
    )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning("Unsupported image content type: %s", file.content_type)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Upload a JPEG, PNG, or BMP image.",
        )

    contents = await file.read()
    await file.close()
    if not contents:
        logger.warning("Empty image upload: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image is empty.",
        )

    try:
        with Image.open(BytesIO(contents)) as image:
            image.load()
            result = predictor.predict(image)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        logger.warning("Invalid image upload: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded content could not be decoded as an image.",
        ) from None
    except Exception:
        logger.exception("Unexpected inference error for %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the image.",
        ) from None

    logger.info(
        "Prediction complete: class=%s, confidence=%.6f",
        result["predicted_class"],
        result["confidence"],
    )
    return result
