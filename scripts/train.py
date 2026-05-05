#!/usr/bin/env python
"""Train one APTOS fold."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader

from dr_grading.config import load_config
from dr_grading.data.dataset import RetinopathyDataset, build_weighted_sampler, compute_class_weights
from dr_grading.data.folds import split_fold
from dr_grading.data.transforms import build_train_transforms, build_valid_transforms
from dr_grading.models.architectures import (
    BaselineEfficientNetClassifier,
    OrdinalBackboneModel,
    RegressionBackboneModel,
)
from dr_grading.training.losses import build_loss_fn
from dr_grading.training.trainer import train_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--fold-csv", type=Path, default=Path("data/interim/aptos2019_folds.csv"))
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def build_model(task: str, arch: str, pretrained: bool, num_classes: int, dropout: float) -> torch.nn.Module:
    if task == "classification":
        return BaselineEfficientNetClassifier(
            arch=arch,
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout,
        )
    if task == "regression":
        return RegressionBackboneModel(arch=arch, pretrained=pretrained, dropout=dropout)
    if task == "ordinal":
        return OrdinalBackboneModel(
            arch=arch,
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported task: {task}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    fold = config.training.fold if args.fold is None else args.fold
    folded = pd.read_csv(args.fold_csv)
    train_df, valid_df = split_fold(folded, fold=fold)

    image_size = config.preprocessing.image_size
    processed_dir = args.processed_dir or (
        config.data.processed_dir / "train" / f"{image_size}px" / "ben_graham"
    )
    image_dir = processed_dir if processed_dir.exists() else config.data.train_image_dir

    train_dataset = RetinopathyDataset(
        train_df,
        image_dir=image_dir,
        transforms=build_train_transforms(image_size),
        task=config.model.task,
        num_classes=config.data.num_classes,
    )
    valid_dataset = RetinopathyDataset(
        valid_df,
        image_dir=image_dir,
        transforms=build_valid_transforms(image_size),
        task=config.model.task,
        num_classes=config.data.num_classes,
    )

    sampler = (
        build_weighted_sampler(train_df["diagnosis"], num_classes=config.data.num_classes)
        if args.use_weighted_sampler
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        task=config.model.task,
        arch=config.model.architecture,
        pretrained=config.model.pretrained,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout,
    )

    class_weights = None
    if config.model.task == "classification":
        class_weights = compute_class_weights(train_df["diagnosis"], config.data.num_classes).to(device)
    loss_fn = build_loss_fn(task=config.model.task, class_weights=class_weights, label_smoothing=0.02)

    if not args.no_wandb:
        wandb.init(
            project=config.project.name,
            name=f"{config.model.architecture}-fold{fold}-{config.model.task}",
            config={"fold": fold, "config_path": str(args.config)},
        )

    checkpoint_path = Path("models/checkpoints") / f"{config.model.architecture}_fold{fold}.pth"
    checkpoint_config: dict[str, Any] = {
        "architecture": config.model.architecture,
        "task": config.model.task,
        "image_size": image_size,
        "num_classes": config.data.num_classes,
    }
    result = train_fold(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_fn=loss_fn,
        device=device,
        epochs=config.training.epochs,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        warmup_epochs=config.training.warmup_epochs,
        gradient_clip_norm=config.training.gradient_clip_norm,
        early_stopping_patience=config.training.early_stopping_patience,
        checkpoint_path=checkpoint_path,
        task=config.model.task,
        config_for_checkpoint=checkpoint_config,
        use_wandb=not args.no_wandb,
        mixed_precision=config.training.mixed_precision,
    )
    print(f"Best fold {fold} QWK={result.best_qwk:.5f} at epoch {result.best_epoch}")


if __name__ == "__main__":
    main()
