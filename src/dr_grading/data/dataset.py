"""PyTorch datasets and samplers for APTOS diabetic retinopathy grading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

TaskMode = Literal["classification", "regression", "ordinal"]


@dataclass(frozen=True)
class DatasetItem:
    image: torch.Tensor
    target: torch.Tensor
    image_id: str
    path: str


class RetinopathyDataset(Dataset[dict[str, Any]]):
    """Dataset for processed or raw fundus images with optional augmentations."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        image_extension: str = "png",
        transforms: Any | None = None,
        task: TaskMode = "classification",
        num_classes: int = 5,
        path_col: str | None = None,
        target_col: str = "diagnosis",
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.image_dir = image_dir
        self.image_extension = image_extension
        self.transforms = transforms
        self.task = task
        self.num_classes = num_classes
        self.path_col = path_col
        self.target_col = target_col

        if "id_code" not in self.dataframe.columns:
            raise ValueError("dataframe must contain id_code")
        if self.target_col not in self.dataframe.columns:
            raise ValueError(f"dataframe must contain {self.target_col}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def _image_path(self, row: pd.Series) -> Path:
        if self.path_col and self.path_col in row and pd.notna(row[self.path_col]):
            return Path(str(row[self.path_col]))
        return self.image_dir / f"{row['id_code']}.{self.image_extension}"

    def _read_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _encode_target(self, label: int) -> torch.Tensor:
        if self.task == "classification":
            return torch.tensor(label, dtype=torch.long)
        if self.task == "regression":
            return torch.tensor([float(label)], dtype=torch.float32)
        if self.task == "ordinal":
            thresholds = torch.arange(self.num_classes - 1, dtype=torch.float32)
            return (thresholds < label).float()
        raise ValueError(f"Unsupported task: {self.task}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        image_id = str(row["id_code"])
        label = int(row[self.target_col])
        path = self._image_path(row)
        image = self._read_image(path)

        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image_tensor = transformed["image"]
        else:
            image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float().div(255.0)

        return {
            "image": image_tensor,
            "target": self._encode_target(label),
            "label": label,
            "image_id": image_id,
            "path": str(path),
        }


def compute_class_weights(labels: pd.Series, num_classes: int = 5) -> torch.Tensor:
    """Compute inverse-frequency class weights normalized around 1.0."""

    counts = labels.astype(int).value_counts().reindex(range(num_classes), fill_value=0).sort_index()
    counts_tensor = torch.tensor(counts.values, dtype=torch.float32)
    weights = counts_tensor.sum() / (num_classes * torch.clamp(counts_tensor, min=1.0))
    return weights / weights.mean()


def build_weighted_sampler(
    labels: pd.Series,
    num_classes: int = 5,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Build a sample-level weighted sampler for imbalanced DR labels."""

    class_weights = compute_class_weights(labels, num_classes=num_classes)
    sample_weights = labels.astype(int).map(lambda label: float(class_weights[int(label)])).to_numpy()
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=replacement,
    )


def mix_external_data(
    aptos_train_df: pd.DataFrame,
    external_df: pd.DataFrame,
    max_external_per_class: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Append optional Kaggle 2015 DR data for pretraining or warm-start training.

    The caller should only use this for pretraining, or for training folds after the
    APTOS validation fold has been fixed. Never let external images influence fold
    assignment or validation threshold optimization.
    """

    required = {"id_code", "diagnosis"}
    missing = required.difference(external_df.columns)
    if missing:
        raise ValueError(f"External dataframe missing required columns: {sorted(missing)}")

    sampled_external = external_df.copy()
    sampled_external["source"] = "kaggle_2015"
    if max_external_per_class is not None:
        sampled_external = (
            sampled_external.groupby("diagnosis", group_keys=False)
            .apply(lambda frame: frame.sample(min(len(frame), max_external_per_class), random_state=seed))
            .reset_index(drop=True)
        )

    aptos = aptos_train_df.copy()
    aptos["source"] = aptos.get("source", "aptos_2019")
    return pd.concat([aptos, sampled_external], ignore_index=True)
