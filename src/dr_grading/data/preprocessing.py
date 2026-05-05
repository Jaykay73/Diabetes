"""Retinal fundus preprocessing used by strong APTOS diabetic retinopathy solutions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from dr_grading.data.quality import read_image_rgb


@dataclass(frozen=True)
class PreprocessOptions:
    image_size: int = 512
    ben_graham_sigma_divisor: int = 30
    circle_crop: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)


def crop_black_border(image: np.ndarray, tolerance: int = 7) -> np.ndarray:
    """Crop dark borders while preserving the retinal field of view."""

    if image.ndim != 3:
        raise ValueError(f"Expected RGB image with 3 dimensions, got shape {image.shape}")

    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = grayscale > tolerance
    if not mask.any():
        return image

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return image[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def resize_keep_aspect(image: np.ndarray, image_size: int) -> np.ndarray:
    """Resize so the longest side equals image_size, then pad to a square."""

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image shape: {image.shape}")

    scale = image_size / max(height, width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    output = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    y0 = (image_size - resized_height) // 2
    x0 = (image_size - resized_width) // 2
    output[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return output


def circle_crop(image: np.ndarray) -> np.ndarray:
    """Mask pixels outside the retinal circle to remove black corner artifacts."""

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    radius = min(center[0], center[1], width - center[0], height - center[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    return cv2.bitwise_and(image, image, mask=mask)


def ben_graham_preprocess(image: np.ndarray, sigma_divisor: int = 30) -> np.ndarray:
    """Apply Ben Graham color normalization: 4*x - 4*blur + 128."""

    if sigma_divisor <= 0:
        raise ValueError("sigma_divisor must be positive")
    sigma = max(image.shape[:2]) / sigma_divisor
    blurred = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma)
    normalized = cv2.addWeighted(image, 4.0, blurred, -4.0, 128.0)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def apply_clahe_rgb(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE on the L channel in LAB space."""

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_lightness = clahe.apply(lightness)
    enhanced = cv2.merge((enhanced_lightness, channel_a, channel_b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def preprocess_fundus_image(
    image: np.ndarray,
    options: PreprocessOptions,
    method: str = "ben_graham",
) -> np.ndarray:
    """Preprocess one RGB fundus image for model training or inference."""

    image = crop_black_border(image)
    image = resize_keep_aspect(image, options.image_size)
    if options.circle_crop:
        image = circle_crop(image)

    if method == "ben_graham":
        return ben_graham_preprocess(image, sigma_divisor=options.ben_graham_sigma_divisor)
    if method == "clahe":
        return apply_clahe_rgb(
            image,
            clip_limit=options.clahe_clip_limit,
            tile_grid_size=options.clahe_tile_grid_size,
        )
    if method == "none":
        return image

    raise ValueError(f"Unsupported preprocessing method: {method}")


def save_rgb_image(image: np.ndarray, path: Path) -> None:
    """Save an RGB uint8 image with OpenCV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise ValueError(f"Could not write image: {path}")


def preprocess_from_csv(
    csv_path: Path,
    image_dir: Path,
    output_dir: Path,
    options: PreprocessOptions,
    method: str,
    image_extension: str = "png",
) -> pd.DataFrame:
    """Preprocess all images referenced by a Kaggle-style id_code CSV."""

    frame = pd.read_csv(csv_path)
    if "id_code" not in frame.columns:
        raise ValueError(f"CSV must contain id_code column: {csv_path}")

    records: list[dict[str, str]] = []
    for image_id in tqdm(frame["id_code"].astype(str), total=len(frame), desc=f"Preprocess {method}"):
        source_path = image_dir / f"{image_id}.{image_extension}"
        target_path = output_dir / method / f"{image_id}.png"
        try:
            image = read_image_rgb(source_path)
            processed = preprocess_fundus_image(image, options=options, method=method)
            save_rgb_image(processed, target_path)
            records.append({"id_code": image_id, "processed_path": str(target_path), "status": "ok"})
        except (ValueError, OSError) as exc:
            records.append({"id_code": image_id, "processed_path": str(target_path), "status": str(exc)})

    manifest = pd.DataFrame.from_records(records)
    manifest_path = output_dir / method / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest
