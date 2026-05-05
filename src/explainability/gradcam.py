"""Grad-CAM utilities for retinal fundus model explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from dr_grading.data.preprocessing import PreprocessOptions, preprocess_fundus_image


@dataclass(frozen=True)
class CAMResult:
    """Container returned by CAM explainers."""

    heatmap: np.ndarray
    overlay_original: Image.Image
    overlay_preprocessed: Image.Image


class RegressionOutputTarget:
    """pytorch-grad-cam target for one-output regression models."""

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim == 2:
            return model_output[:, 0]
        return model_output.reshape(model_output.size(0), -1)[:, 0]


class ClassifierOutputTarget:
    """Small local target class to avoid importing pytorch-grad-cam at module import time."""

    def __init__(self, category: int) -> None:
        self.category = category

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim != 2:
            raise ValueError(f"Expected classifier output [B, C], got {tuple(model_output.shape)}")
        return model_output[:, self.category]


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the innermost timm model when using local wrapper classes."""

    if hasattr(model, "backbone") and isinstance(getattr(model, "backbone"), nn.Module):
        return getattr(model, "backbone")
    return model


def _last_module(module: nn.Module, attr: str) -> nn.Module | None:
    value = getattr(module, attr, None)
    if isinstance(value, nn.Sequential | nn.ModuleList):
        return value[-1]
    if isinstance(value, list | tuple) and value and isinstance(value[-1], nn.Module):
        return value[-1]
    return value if isinstance(value, nn.Module) else None


def get_target_layer(model: nn.Module, arch_name: str) -> nn.Module:
    """Resolve the best final spatial layer for timm EfficientNet and ConvNeXt models.

    Supported examples:
    - `tf_efficientnet_b4_ns`, `efficientnet_b4`, `tf_efficientnet_b5_ns`
    - `convnext_large`, `convnext_large_in22k`
    - `tf_efficientnetv2_m`

    The function also handles the local wrapper classes from `dr_grading.models`.
    """

    arch = arch_name.lower()
    candidate = _unwrap_model(model)

    if "convnext" in arch:
        stages = _last_module(candidate, "stages")
        if stages is not None:
            blocks = _last_module(stages, "blocks")
            return blocks or stages
        if hasattr(candidate, "blocks"):
            return getattr(candidate, "blocks")[-1]

    if "efficientnetv2" in arch or "efficientnet" in arch:
        blocks = _last_module(candidate, "blocks")
        if blocks is not None:
            return blocks
        features = _last_module(candidate, "features")
        if features is not None:
            return features
        conv_head = getattr(candidate, "conv_head", None)
        if isinstance(conv_head, nn.Module):
            return conv_head

    features = _last_module(candidate, "features")
    if features is not None:
        return features
    blocks = _last_module(candidate, "blocks")
    if blocks is not None:
        return blocks

    raise ValueError(
        f"Could not infer Grad-CAM target layer for architecture '{arch_name}'. "
        "Pass a target_layer explicitly."
    )


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalize a heatmap to float32 [0, 1]."""

    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = heatmap - float(heatmap.min())
    denominator = float(heatmap.max())
    if denominator <= 1e-8:
        return np.zeros_like(heatmap, dtype=np.float32)
    return (heatmap / denominator).astype(np.float32)


def overlay_cam(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    """Overlay a CAM heatmap on an RGB image."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H, W, 3], got {image.shape}")
    resized_heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * normalize_heatmap(resized_heatmap))
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image.astype(np.uint8), 1.0 - alpha, colored, alpha, 0.0)
    return Image.fromarray(overlay)


class GradCAMExplainer:
    """Wrapper around `pytorch-grad-cam` returning clinical-review friendly artifacts."""

    def __init__(
        self,
        model: nn.Module,
        arch_name: str,
        target_layer: nn.Module | None = None,
        reshape_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.arch_name = arch_name
        self.target_layer = target_layer or get_target_layer(model, arch_name)
        self.reshape_transform = reshape_transform
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _build_cam(self) -> object:
        try:
            from pytorch_grad_cam import GradCAM
        except ImportError as exc:
            raise ImportError(
                "pytorch-grad-cam is required for GradCAMExplainer. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        return GradCAM(
            model=self.model,
            target_layers=[self.target_layer],
            reshape_transform=self.reshape_transform,
        )

    def explain(
        self,
        image_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_category: int | None = None,
        preprocessed_image: np.ndarray | None = None,
        regression: bool = False,
        opacity: float = 0.35,
    ) -> CAMResult:
        """Generate a Grad-CAM heatmap and overlays.

        Args:
            image_tensor: normalized tensor shaped `[3, H, W]` or `[1, 3, H, W]`.
            original_image: original RGB image as uint8 `[H, W, 3]`.
            target_category: class index for classifier CAMs.
            preprocessed_image: optional Ben-Graham image; computed if omitted.
            regression: use `RegressionOutputTarget` for one-output regression models.
            opacity: heatmap overlay opacity.
        """

        if image_tensor.ndim == 3:
            input_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim == 4 and image_tensor.size(0) == 1:
            input_tensor = image_tensor
        else:
            raise ValueError(f"Expected image tensor [3,H,W] or [1,3,H,W], got {tuple(image_tensor.shape)}")

        input_tensor = input_tensor.to(self.device)
        if regression:
            targets = [RegressionOutputTarget()]
        elif target_category is not None:
            targets = [ClassifierOutputTarget(target_category)]
        else:
            targets = None

        with self._build_cam() as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        heatmap = normalize_heatmap(grayscale_cam)
        if preprocessed_image is None:
            preprocessed_image = preprocess_fundus_image(
                original_image,
                options=PreprocessOptions(image_size=original_image.shape[0]),
                method="ben_graham",
            )

        return CAMResult(
            heatmap=heatmap.astype(np.float32),
            overlay_original=overlay_cam(original_image, heatmap, alpha=opacity),
            overlay_preprocessed=overlay_cam(preprocessed_image, heatmap, alpha=opacity),
        )


CAMMethod = Literal["gradcam", "gradcam++", "eigencam", "scorecam", "layercam", "ablationcam"]


class CAMVariantExplainer(GradCAMExplainer):
    """CAM explainer supporting common pytorch-grad-cam variants."""

    def __init__(
        self,
        model: nn.Module,
        arch_name: str,
        method: CAMMethod = "gradcam",
        target_layer: nn.Module | None = None,
        reshape_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            model=model,
            arch_name=arch_name,
            target_layer=target_layer,
            reshape_transform=reshape_transform,
            device=device,
        )
        self.method = method

    def _build_cam(self) -> object:
        try:
            from pytorch_grad_cam import (
                AblationCAM,
                EigenCAM,
                GradCAM,
                GradCAMPlusPlus,
                LayerCAM,
                ScoreCAM,
            )
        except ImportError as exc:
            raise ImportError(
                "grad-cam is required for CAMVariantExplainer. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        registry = {
            "gradcam": GradCAM,
            "gradcam++": GradCAMPlusPlus,
            "eigencam": EigenCAM,
            "scorecam": ScoreCAM,
            "layercam": LayerCAM,
            "ablationcam": AblationCAM,
        }
        cam_cls = registry[self.method]
        return cam_cls(
            model=self.model,
            target_layers=[self.target_layer],
            reshape_transform=self.reshape_transform,
        )


def build_multiclass_cam_grid(
    explainer: GradCAMExplainer,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    preprocessed_image: np.ndarray,
    opacity: float = 0.35,
) -> Image.Image:
    """Generate a 1x5 grid of CAM overlays for grades 0 through 4."""

    overlays: list[np.ndarray] = []
    for grade in range(5):
        result = explainer.explain(
            image_tensor=image_tensor,
            original_image=original_image,
            preprocessed_image=preprocessed_image,
            target_category=grade,
            regression=False,
            opacity=opacity,
        )
        overlays.append(np.asarray(result.overlay_preprocessed))

    height, width = overlays[0].shape[:2]
    title_height = max(28, height // 12)
    canvas = np.full((height + title_height, width * 5, 3), 255, dtype=np.uint8)
    for grade, overlay in enumerate(overlays):
        x0 = grade * width
        canvas[title_height:, x0 : x0 + width] = overlay
        cv2.putText(
            canvas,
            f"Grade {grade}",
            (x0 + 10, max(20, title_height - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.45, width / 600),
            (20, 20, 20),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return Image.fromarray(canvas)
