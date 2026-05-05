#!/usr/bin/env python
"""Create a Stratified K-Fold CSV for APTOS training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dr_grading.config import load_config
from dr_grading.data.folds import save_stratified_folds
from dr_grading.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/aptos2019_folds.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    logger = logging.getLogger(__name__)
    config = load_config(args.config)
    folded = save_stratified_folds(
        train_csv=config.data.train_csv,
        output_csv=args.output,
        n_splits=config.training.folds,
        seed=config.project.seed,
    )
    logger.info("Saved %d rows with %d folds to %s", len(folded), config.training.folds, args.output)


if __name__ == "__main__":
    main()
