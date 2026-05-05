#!/usr/bin/env python
"""Run Phase 1 EDA for the APTOS 2019 dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dr_grading.config import load_config
from dr_grading.data.eda import run_phase1_eda
from dr_grading.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    logger = logging.getLogger(__name__)
    config = load_config(args.config)
    outputs = run_phase1_eda(config)
    for name, path in outputs.items():
        logger.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
