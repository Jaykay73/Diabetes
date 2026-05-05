"""Deep SHAP helpers for CNN image models."""

from __future__ import annotations

import numpy as np
import torch


def build_shap_background(loader, per_class: int = 50, device: torch.device | None = None) -> torch.Tensor:
    """Sample a roughly class-balanced SHAP background tensor."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buckets: dict[int, list[torch.Tensor]] = {idx: [] for idx in range(5)}
    for batch in loader:
        labels = batch["label"].tolist()
        for image, label in zip(batch["image"], labels):
            if len(buckets[int(label)]) < per_class:
                buckets[int(label)].append(image)
        if all(len(items) >= per_class for items in buckets.values()):
            break
    images = [image for items in buckets.values() for image in items]
    if not images:
        raise ValueError("Could not sample SHAP background from loader")
    return torch.stack(images).to(device)


def compute_deep_shap_values(model: torch.nn.Module, background: torch.Tensor, images: torch.Tensor):
    """Compute SHAP values with shap.DeepExplainer."""

    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install shap to use compute_deep_shap_values.") from exc

    model.eval()
    explainer = shap.DeepExplainer(model, background)
    return explainer.shap_values(images.to(background.device))


def mean_abs_shap_grid(shap_values: np.ndarray, grid_size: int = 8) -> np.ndarray:
    """Aggregate absolute SHAP values into an HxW region grid."""

    values = np.asarray(shap_values)
    if values.ndim == 4 and values.shape[1] in {1, 3}:
        values = np.transpose(values, (0, 2, 3, 1))
    if values.ndim == 3:
        values = values[None, ...]
    h, w = values.shape[1:3]
    region_h = max(1, h // grid_size)
    region_w = max(1, w // grid_size)
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    abs_values = np.abs(values).mean(axis=(0, 3))
    for row in range(grid_size):
        for col in range(grid_size):
            patch = abs_values[row * region_h : (row + 1) * region_h, col * region_w : (col + 1) * region_w]
            grid[row, col] = float(patch.mean()) if patch.size else 0.0
    return grid
