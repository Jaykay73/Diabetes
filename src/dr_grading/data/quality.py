"""Image quality checks for retinal fundus datasets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


@dataclass(frozen=True)
class QualityThresholds:
    black_mean_threshold: float
    black_std_threshold: float
    low_contrast_std_threshold: float
    duplicate_hash_size: int


def read_image_rgb(path: Path) -> np.ndarray:
    """Read an image as RGB uint8 and raise a clear error for corrupt files."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_stats(image: np.ndarray) -> dict[str, float | int]:
    """Compute basic image statistics used for EDA and quality flags."""

    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = grayscale.shape
    return {
        "height": int(height),
        "width": int(width),
        "mean": float(grayscale.mean()),
        "std": float(grayscale.std()),
        "min": int(grayscale.min()),
        "max": int(grayscale.max()),
    }


def flag_quality_issues(stats: dict[str, float | int], thresholds: QualityThresholds) -> list[str]:
    """Assign deterministic quality issue flags from image statistics."""

    flags: list[str] = []
    mean = float(stats["mean"])
    std = float(stats["std"])
    if mean <= thresholds.black_mean_threshold and std <= thresholds.black_std_threshold:
        flags.append("black_or_nearly_black")
    if std <= thresholds.low_contrast_std_threshold:
        flags.append("low_contrast")
    return flags


def perceptual_hash(path: Path, hash_size: int) -> str:
    """Compute a perceptual hash for duplicate and near-duplicate discovery."""

    try:
        with Image.open(path) as image:
            return str(imagehash.phash(image.convert("RGB"), hash_size=hash_size))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not hash image: {path}") from exc


def build_image_quality_report(
    labels_df: pd.DataFrame,
    image_dir: Path,
    thresholds: QualityThresholds,
    image_extension: str = "png",
) -> pd.DataFrame:
    """Create one row per image with dimensions, quality flags, and pHash."""

    records: list[dict[str, object]] = []
    hash_to_ids: dict[str, list[str]] = defaultdict(list)

    for row in tqdm(labels_df.itertuples(index=False), total=len(labels_df), desc="Quality audit"):
        image_id = str(row.id_code)
        diagnosis = int(row.diagnosis)
        image_path = image_dir / f"{image_id}.{image_extension}"
        record: dict[str, object] = {
            "id_code": image_id,
            "diagnosis": diagnosis,
            "path": str(image_path),
            "exists": image_path.exists(),
            "error": "",
        }

        if not image_path.exists():
            record.update({"flags": "missing", "phash": ""})
            records.append(record)
            continue

        try:
            image = read_image_rgb(image_path)
            stats = image_stats(image)
            phash = perceptual_hash(image_path, thresholds.duplicate_hash_size)
            flags = flag_quality_issues(stats, thresholds)
            hash_to_ids[phash].append(image_id)
            record.update(stats)
            record.update({"flags": ",".join(flags), "phash": phash})
        except ValueError as exc:
            record.update({"flags": "corrupt_or_unreadable", "phash": "", "error": str(exc)})
        records.append(record)

    report = pd.DataFrame.from_records(records)
    duplicate_hashes = {key for key, values in hash_to_ids.items() if len(values) > 1}
    if not report.empty:
        report["is_duplicate_phash"] = report["phash"].isin(duplicate_hashes)
    return report
