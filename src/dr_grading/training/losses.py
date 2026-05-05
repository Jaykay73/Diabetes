"""Loss construction for classification, regression, and ordinal training."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from dr_grading.models.architectures import coral_loss, smooth_ordinal_targets


def build_loss_fn(
    task: str,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create a loss function matching the model task."""

    if task == "classification":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if task == "regression":
        return nn.MSELoss()
    if task == "ordinal":

        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            smoothed = smooth_ordinal_targets(targets, smoothing=label_smoothing)
            return coral_loss(logits, smoothed)

        return loss_fn
    raise ValueError(f"Unsupported task: {task}")
