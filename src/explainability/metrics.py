"""Faithfulness metrics for image explanations."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import auc


def compute_insertion_deletion_auc(
    model: torch.nn.Module,
    image: torch.Tensor,
    heatmap: np.ndarray,
    steps: int = 20,
) -> dict[str, float]:
    """Compute insertion/deletion AUC for a scalar-output model."""

    model.eval()
    device = next(model.parameters()).device
    image = image.detach().to(device)
    _, height, width = image.shape
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = cv2_resize_like(heatmap, height, width)
    order = np.argsort(heatmap.reshape(-1))[::-1]
    blurred = torch.nn.functional.avg_pool2d(image.unsqueeze(0), 15, stride=1, padding=7).squeeze(0)
    deletion_scores: list[float] = []
    insertion_scores: list[float] = []
    for step in range(steps + 1):
        k = int(len(order) * step / steps)
        mask = torch.zeros(height * width, dtype=torch.bool, device=device)
        mask[torch.as_tensor(order[:k], device=device)] = True
        mask = mask.view(height, width)
        deleted = image.clone()
        inserted = blurred.clone()
        deleted[:, mask] = 0
        inserted[:, mask] = image[:, mask]
        with torch.no_grad():
            deletion_scores.append(float(model(deleted.unsqueeze(0)).reshape(-1)[0].detach().cpu()))
            insertion_scores.append(float(model(inserted.unsqueeze(0)).reshape(-1)[0].detach().cpu()))
    x = np.linspace(0, 1, steps + 1)
    return {"deletion_auc": float(auc(x, deletion_scores)), "insertion_auc": float(auc(x, insertion_scores))}


def cv2_resize_like(heatmap: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    return cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)


def pointing_game_hit(heatmap: np.ndarray, pathology_mask: np.ndarray) -> bool:
    """Return True if the heatmap peak falls inside a pathology mask."""

    y, x = np.unravel_index(np.asarray(heatmap).argmax(), heatmap.shape)
    resized_mask = cv2_resize_like(pathology_mask.astype(np.float32), heatmap.shape[0], heatmap.shape[1])
    return bool(resized_mask[y, x] > 0)


def explanation_stability_score(
    explain_fn,
    image: torch.Tensor,
    n_trials: int = 10,
    noise_std: float = 0.01,
) -> float:
    """Measure heatmap stability under small Gaussian input perturbations."""

    base = np.asarray(explain_fn(image), dtype=np.float32)
    scores: list[float] = []
    for _ in range(n_trials):
        perturbed = image + torch.randn_like(image) * noise_std
        trial = np.asarray(explain_fn(perturbed), dtype=np.float32)
        numerator = float(np.linalg.norm(base - trial))
        denominator = float(np.linalg.norm(base) + 1e-6)
        scores.append(max(0.0, 1.0 - numerator / denominator))
    return float(np.mean(scores))
