"""PyTorch model architectures for APTOS diabetic retinopathy grading."""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import nn
from torch.nn import functional as F


class GeM(nn.Module):
    """Generalized mean pooling, a strong drop-in replacement for average pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = True) -> None:
        super().__init__()
        initial = torch.ones(1) * p
        self.p = nn.Parameter(initial) if trainable else initial
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        return F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1))).pow(1.0 / self.p)


class BaselineEfficientNetClassifier(nn.Module):
    """Approach A: 5-class EfficientNet-B4 classifier."""

    def __init__(
        self,
        arch: str = "tf_efficientnet_b4_ns",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.backbone = timm.create_model(
            arch,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_rate=dropout,
        )
        in_features = int(self.backbone.num_features)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class RegressionBackboneModel(nn.Module):
    """Approach B: ordinal label as a continuous regression target in [0, 4]."""

    def __init__(
        self,
        arch: str = "tf_efficientnet_b4_ns",
        pretrained: bool = True,
        dropout: float = 0.3,
        output_min: float = 0.0,
        output_max: float = 4.0,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.output_min = output_min
        self.output_max = output_max
        self.backbone = timm.create_model(
            arch,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_rate=dropout,
        )
        in_features = int(self.backbone.num_features)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        raw = self.regressor(features)
        return raw.clamp(self.output_min, self.output_max)


@dataclass(frozen=True)
class OrdinalOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    expected_grade: torch.Tensor


class OrdinalBackboneModel(nn.Module):
    """Approach C: stronger timm backbone with GeM pooling and ordinal thresholds."""

    def __init__(
        self,
        arch: str = "tf_efficientnet_b5_ns",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.35,
        use_gem: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2 for ordinal regression")

        self.arch = arch
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            arch,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),
            drop_rate=dropout,
        )
        feature_info = self.backbone.feature_info[-1]
        in_channels = int(feature_info["num_chs"])
        self.pool = GeM() if use_gem else nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes - 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(x)[-1]
        pooled = self.pool(feature_map)
        return self.head(pooled)

    def predict_proba(self, x: torch.Tensor) -> OrdinalOutput:
        """Convert ordered-threshold logits into class probabilities."""

        logits = self.forward(x)
        threshold_probs = torch.sigmoid(logits)
        batch_size = threshold_probs.size(0)
        probabilities = torch.zeros(
            batch_size,
            self.num_classes,
            device=threshold_probs.device,
            dtype=threshold_probs.dtype,
        )
        probabilities[:, 0] = 1.0 - threshold_probs[:, 0]
        for class_idx in range(1, self.num_classes - 1):
            probabilities[:, class_idx] = (
                threshold_probs[:, class_idx - 1] - threshold_probs[:, class_idx]
            )
        probabilities[:, self.num_classes - 1] = threshold_probs[:, self.num_classes - 2]
        probabilities = probabilities.clamp(min=0.0, max=1.0)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp(min=1e-6)
        grades = torch.arange(self.num_classes, device=x.device, dtype=probabilities.dtype)
        expected_grade = probabilities.matmul(grades)
        return OrdinalOutput(logits=logits, probabilities=probabilities, expected_grade=expected_grade)


def coral_loss(logits: torch.Tensor, levels: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Binary cross-entropy over ordered class thresholds."""

    loss = F.binary_cross_entropy_with_logits(logits, levels.float(), reduction="none").sum(dim=1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")


def smooth_ordinal_targets(levels: torch.Tensor, smoothing: float = 0.02) -> torch.Tensor:
    """Apply light label smoothing to ordinal threshold targets."""

    if not 0.0 <= smoothing < 0.5:
        raise ValueError("smoothing must be in [0, 0.5)")
    return levels.float() * (1.0 - smoothing) + (1.0 - levels.float()) * smoothing
