#!/usr/bin/env python
"""Generate a Kaggle submission CSV from checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from dr_grading.inference.predictor import CheckpointSpec, EnsemblePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", type=Path, default=Path("data/raw/aptos2019/test.csv"))
    parser.add_argument("--test-image-dir", type=Path, default=Path("data/raw/aptos2019/test_images"))
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--architecture", action="append", required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--weight", type=float, action="append", default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--thresholds", type=float, nargs=4, default=None)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weight or [1.0] * len(args.checkpoint)
    if not (len(args.checkpoint) == len(args.architecture) == len(args.task) == len(weights)):
        raise ValueError("checkpoint, architecture, task, and weight counts must match")

    specs = [
        CheckpointSpec(path=path, architecture=arch, task=task, weight=weight)
        for path, arch, task, weight in zip(args.checkpoint, args.architecture, args.task, weights)
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = EnsemblePredictor(specs, device=device, image_size=args.image_size)
    test_df = pd.read_csv(args.test_csv)

    rows: list[dict[str, int | str]] = []
    for image_id in tqdm(test_df["id_code"].astype(str), desc="Predict test"):
        image_path = args.test_image_dir / f"{image_id}.png"
        with Image.open(image_path) as image:
            result = predictor.predict_grade(image.convert("RGB"), thresholds=args.thresholds)
        rows.append({"id_code": image_id, "diagnosis": int(result["grade"])})

    submission = pd.DataFrame(rows)
    submission.to_csv(args.output, index=False)
    print(f"Saved {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
