"""Model loading, TTA, and ensembling."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from dr_grading.data.preprocessing import PreprocessOptions, preprocess_fundus_image
from dr_grading.data.transforms import build_valid_transforms
from dr_grading.models.architectures import (
    BaselineEfficientNetClassifier,
    OrdinalBackboneModel,
    RegressionBackboneModel,
)
from dr_grading.training.metrics import ordinal_logits_to_expected_grade


SEVERITY_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


@dataclass(frozen=True)
class CheckpointSpec:
    path: Path
    architecture: str
    task: str
    weight: float = 1.0
    num_classes: int = 5
    dropout: float = 0.0


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to RGB uint8 numpy."""

    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def image_to_base64(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_pil(payload: str) -> Image.Image:
    try:
        raw = base64.b64decode(payload)
        return Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid base64 image payload") from exc


def build_inference_model(spec: CheckpointSpec) -> nn.Module:
    """Construct a model matching a checkpoint specification."""

    if spec.task == "classification":
        return BaselineEfficientNetClassifier(
            arch=spec.architecture,
            pretrained=False,
            num_classes=spec.num_classes,
            dropout=spec.dropout,
        )
    if spec.task == "regression":
        return RegressionBackboneModel(
            arch=spec.architecture,
            pretrained=False,
            dropout=spec.dropout,
        )
    if spec.task == "ordinal":
        return OrdinalBackboneModel(
            arch=spec.architecture,
            pretrained=False,
            num_classes=spec.num_classes,
            dropout=spec.dropout,
        )
    raise ValueError(f"Unsupported task: {spec.task}")


def load_checkpoint_model(spec: CheckpointSpec, device: torch.device) -> nn.Module:
    """Load a trained checkpoint onto device."""

    checkpoint = torch.load(spec.path, map_location=device)
    model = build_inference_model(spec)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _tta_batch(image_tensor: torch.Tensor) -> torch.Tensor:
    """Create original, horizontal flip, and vertical flip TTA batch."""

    return torch.stack(
        [
            image_tensor,
            torch.flip(image_tensor, dims=[2]),
            torch.flip(image_tensor, dims=[1]),
        ],
        dim=0,
    )


def _continuous_prediction(task: str, outputs: torch.Tensor) -> torch.Tensor:
    if task == "classification":
        probabilities = torch.softmax(outputs, dim=1)
        grades = torch.arange(probabilities.size(1), device=outputs.device, dtype=probabilities.dtype)
        return probabilities.matmul(grades)
    if task == "regression":
        return outputs.reshape(-1).clamp(0.0, 4.0)
    if task == "ordinal":
        values = ordinal_logits_to_expected_grade(outputs.detach().cpu().numpy())
        return torch.as_tensor(values, device=outputs.device, dtype=torch.float32)
    raise ValueError(f"Unsupported task: {task}")


@torch.no_grad()
def predict_tensor_tta(model: nn.Module, image_tensor: torch.Tensor, task: str, device: torch.device) -> float:
    """Predict one image using 3-view TTA and return a continuous grade."""

    batch = _tta_batch(image_tensor).to(device)
    outputs = model(batch)
    continuous = _continuous_prediction(task, outputs)
    return float(continuous.mean().detach().cpu())


class EnsemblePredictor:
    """Weighted checkpoint ensemble for DR grade prediction."""

    def __init__(
        self,
        checkpoint_specs: list[CheckpointSpec],
        device: torch.device,
        image_size: int = 512,
        preprocess_method: str = "ben_graham",
    ) -> None:
        if not checkpoint_specs:
            raise ValueError("At least one checkpoint is required")
        self.specs = checkpoint_specs
        self.device = device
        self.image_size = image_size
        self.preprocess_method = preprocess_method
        self.transforms = build_valid_transforms(image_size)
        self.models = [load_checkpoint_model(spec, device=device) for spec in checkpoint_specs]

    def prepare_image(self, image: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
        raw = pil_to_rgb_array(image)
        processed = preprocess_fundus_image(
            raw,
            options=PreprocessOptions(image_size=self.image_size),
            method=self.preprocess_method,
        )
        tensor = self.transforms(image=processed)["image"]
        return processed, tensor

    def predict_continuous(self, image: Image.Image) -> float:
        _, tensor = self.prepare_image(image)
        weighted_sum = 0.0
        total_weight = 0.0
        for model, spec in zip(self.models, self.specs):
            weighted_sum += spec.weight * predict_tensor_tta(model, tensor, spec.task, self.device)
            total_weight += spec.weight
        return weighted_sum / max(total_weight, 1e-8)

    def predict_grade(self, image: Image.Image, thresholds: list[float] | None = None) -> dict[str, Any]:
        continuous = self.predict_continuous(image)
        if thresholds is None:
            grade = int(np.rint(np.clip(continuous, 0.0, 4.0)))
        else:
            grade = int(np.digitize([continuous], bins=np.asarray(thresholds))[0])
        confidence = 1.0 - min(abs(continuous - grade), 1.0)
        return {
            "grade": grade,
            "continuous_prediction": continuous,
            "confidence": float(confidence),
            "severity": SEVERITY_LABELS[grade],
        }


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    """Overlay a normalized heatmap on an RGB image."""

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0)
    return Image.fromarray(overlay)
