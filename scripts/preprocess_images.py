#!/usr/bin/env python
"""Preprocess APTOS images to disk for faster training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dr_grading.config import load_config
from dr_grading.data.preprocessing import PreprocessOptions, preprocess_from_csv
from dr_grading.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--method", choices=["ben_graham", "clahe", "none"], default="ben_graham")
    parser.add_argument("--image-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    logger = logging.getLogger(__name__)
    config = load_config(args.config)

    options = PreprocessOptions(
        image_size=args.image_size or config.preprocessing.image_size,
        ben_graham_sigma_divisor=config.preprocessing.ben_graham_sigma_divisor,
        circle_crop=config.preprocessing.circle_crop,
        clahe_clip_limit=config.preprocessing.clahe_clip_limit,
        clahe_tile_grid_size=config.preprocessing.clahe_tile_grid_size,
    )

    csv_path = config.data.train_csv if args.split == "train" else config.data.test_csv
    image_dir = config.data.train_image_dir if args.split == "train" else config.data.test_image_dir
    output_dir = config.data.processed_dir / args.split / f"{options.image_size}px"

    manifest = preprocess_from_csv(
        csv_path=csv_path,
        image_dir=image_dir,
        output_dir=output_dir,
        options=options,
        method=args.method,
        image_extension=config.data.image_extension,
    )
    failed = manifest[manifest["status"] != "ok"]
    logger.info("Processed %d images into %s", len(manifest) - len(failed), output_dir)
    if not failed.empty:
        logger.warning("Failed to process %d images. See manifest.csv for details.", len(failed))


if __name__ == "__main__":
    main()
