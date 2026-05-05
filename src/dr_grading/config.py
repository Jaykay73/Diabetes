"""Typed configuration loading for the diabetic retinopathy pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int
    output_dir: Path


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    train_csv: Path
    test_csv: Path
    train_image_dir: Path
    test_image_dir: Path
    processed_dir: Path
    external_2015_dir: Path
    image_extension: str
    num_classes: int


@dataclass(frozen=True)
class EDAConfig:
    figures_dir: Path
    quality_dir: Path
    duplicate_hash_size: int
    black_mean_threshold: float
    black_std_threshold: float
    low_contrast_std_threshold: float
    sample_images_per_class: int


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: int
    ben_graham_sigma_divisor: int
    circle_crop: bool
    clahe_clip_limit: float
    clahe_tile_grid_size: tuple[int, int]


@dataclass(frozen=True)
class TrainingConfig:
    folds: int
    fold: int
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    gradient_clip_norm: float
    mixed_precision: bool
    early_stopping_patience: int
    metric: str


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    pretrained: bool
    num_classes: int
    dropout: float
    task: str


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    data: DataConfig
    eda: EDAConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    model: ModelConfig


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def load_config(path: str | Path = "configs/config.yaml") -> AppConfig:
    """Load the project YAML config into immutable dataclasses."""

    raw = _load_yaml(_as_path(path))
    project = raw["project"]
    data = raw["data"]
    eda = raw["eda"]
    preprocessing = raw["preprocessing"]
    training = raw["training"]
    model = raw["model"]

    return AppConfig(
        project=ProjectConfig(
            name=str(project["name"]),
            seed=int(project["seed"]),
            output_dir=_as_path(project["output_dir"]),
        ),
        data=DataConfig(
            raw_dir=_as_path(data["raw_dir"]),
            train_csv=_as_path(data["train_csv"]),
            test_csv=_as_path(data["test_csv"]),
            train_image_dir=_as_path(data["train_image_dir"]),
            test_image_dir=_as_path(data["test_image_dir"]),
            processed_dir=_as_path(data["processed_dir"]),
            external_2015_dir=_as_path(data["external_2015_dir"]),
            image_extension=str(data["image_extension"]),
            num_classes=int(data["num_classes"]),
        ),
        eda=EDAConfig(
            figures_dir=_as_path(eda["figures_dir"]),
            quality_dir=_as_path(eda["quality_dir"]),
            duplicate_hash_size=int(eda["duplicate_hash_size"]),
            black_mean_threshold=float(eda["black_mean_threshold"]),
            black_std_threshold=float(eda["black_std_threshold"]),
            low_contrast_std_threshold=float(eda["low_contrast_std_threshold"]),
            sample_images_per_class=int(eda["sample_images_per_class"]),
        ),
        preprocessing=PreprocessingConfig(
            image_size=int(preprocessing["image_size"]),
            ben_graham_sigma_divisor=int(preprocessing["ben_graham_sigma_divisor"]),
            circle_crop=bool(preprocessing["circle_crop"]),
            clahe_clip_limit=float(preprocessing["clahe_clip_limit"]),
            clahe_tile_grid_size=tuple(preprocessing["clahe_tile_grid_size"]),
        ),
        training=TrainingConfig(
            folds=int(training["folds"]),
            fold=int(training["fold"]),
            batch_size=int(training["batch_size"]),
            num_workers=int(training["num_workers"]),
            epochs=int(training["epochs"]),
            lr=float(training["lr"]),
            weight_decay=float(training["weight_decay"]),
            warmup_epochs=int(training["warmup_epochs"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            mixed_precision=bool(training["mixed_precision"]),
            early_stopping_patience=int(training["early_stopping_patience"]),
            metric=str(training["metric"]),
        ),
        model=ModelConfig(
            architecture=str(model["architecture"]),
            pretrained=bool(model["pretrained"]),
            num_classes=int(model["num_classes"]),
            dropout=float(model["dropout"]),
            task=str(model["task"]),
        ),
    )
