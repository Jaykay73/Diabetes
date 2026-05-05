"""Fold training loop for diabetic retinopathy grading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dr_grading.training.metrics import (
    logits_to_classes,
    ordinal_logits_to_expected_grade,
    quadratic_weighted_kappa,
    regression_to_classes,
)
from dr_grading.training.schedulers import build_warmup_cosine_scheduler


@dataclass(frozen=True)
class TrainResult:
    best_qwk: float
    best_epoch: int
    checkpoint_path: Path


class EarlyStopping:
    """Track validation QWK and stop after patience epochs without improvement."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_score = -np.inf
        self.best_epoch = -1
        self.bad_epochs = 0

    def step(self, score: float, epoch: int) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    targets = batch["target"].to(device, non_blocking=True)
    return images, targets


def _predictions_for_task(task: str, outputs: torch.Tensor) -> np.ndarray:
    values = outputs.detach().float().cpu().numpy()
    if task == "classification":
        return logits_to_classes(values)
    if task == "regression":
        return regression_to_classes(values)
    if task == "ordinal":
        expected_grade = ordinal_logits_to_expected_grade(values)
        return regression_to_classes(expected_grade)
    raise ValueError(f"Unsupported task: {task}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    task: str,
    scaler: torch.cuda.amp.GradScaler,
    gradient_clip_norm: float,
) -> dict[str, float]:
    """Run one mixed-precision training epoch."""

    model.train()
    losses: list[float] = []
    all_true: list[int] = []
    all_pred: list[int] = []

    for batch in tqdm(loader, desc="train", leave=False):
        images, targets = _move_batch(batch, device)
        labels = batch["label"].detach().cpu().numpy().astype(int).tolist()
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(images)
            loss = loss_fn(outputs, targets)

        scaler.scale(loss).backward()
        if gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.append(float(loss.detach().cpu()))
        all_true.extend(labels)
        all_pred.extend(_predictions_for_task(task, outputs).tolist())

    return {
        "loss": float(np.mean(losses)),
        "qwk": quadratic_weighted_kappa(np.asarray(all_true), np.asarray(all_pred)),
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Any,
    device: torch.device,
    task: str,
) -> dict[str, float]:
    """Run validation and compute QWK."""

    model.eval()
    losses: list[float] = []
    all_true: list[int] = []
    all_pred: list[int] = []

    for batch in tqdm(loader, desc="valid", leave=False):
        images, targets = _move_batch(batch, device)
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        labels = batch["label"].detach().cpu().numpy().astype(int).tolist()
        losses.append(float(loss.detach().cpu()))
        all_true.extend(labels)
        all_pred.extend(_predictions_for_task(task, outputs).tolist())

    return {
        "loss": float(np.mean(losses)),
        "qwk": quadratic_weighted_kappa(np.asarray(all_true), np.asarray(all_pred)),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    qwk: float,
    config: dict[str, Any],
) -> None:
    """Save a fold checkpoint with enough metadata for inference."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "qwk": qwk,
            "config": config,
        },
        path,
    )


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    loss_fn: Any,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    gradient_clip_norm: float,
    early_stopping_patience: int,
    checkpoint_path: Path,
    task: str,
    config_for_checkpoint: dict[str, Any],
    use_wandb: bool = True,
    mixed_precision: bool = True,
) -> TrainResult:
    """Train one fold and keep the checkpoint with best validation QWK."""

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = len(train_loader) * warmup_epochs
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision and device.type == "cuda")
    stopper = EarlyStopping(patience=early_stopping_patience)

    best_qwk = -np.inf
    best_epoch = -1
    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            task=task,
            scaler=scaler,
            gradient_clip_norm=gradient_clip_norm,
        )
        valid_metrics = validate_one_epoch(
            model=model,
            loader=valid_loader,
            loss_fn=loss_fn,
            device=device,
            task=task,
        )

        metrics = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/qwk": train_metrics["qwk"],
            "valid/loss": valid_metrics["loss"],
            "valid/qwk": valid_metrics["qwk"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        if use_wandb:
            wandb.log(metrics)

        if valid_metrics["qwk"] > best_qwk:
            best_qwk = valid_metrics["qwk"]
            best_epoch = epoch
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                qwk=best_qwk,
                config=config_for_checkpoint,
            )

        if stopper.step(valid_metrics["qwk"], epoch):
            break

    return TrainResult(best_qwk=float(best_qwk), best_epoch=best_epoch, checkpoint_path=checkpoint_path)
