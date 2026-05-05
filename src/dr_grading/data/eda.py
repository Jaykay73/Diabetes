"""Exploratory data analysis for APTOS diabetic retinopathy images."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from dr_grading.config import AppConfig
from dr_grading.data.quality import QualityThresholds, build_image_quality_report


CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


def load_train_labels(csv_path: Path) -> pd.DataFrame:
    """Load and validate APTOS train labels."""

    labels = pd.read_csv(csv_path)
    expected = {"id_code", "diagnosis"}
    missing = expected.difference(labels.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    labels["diagnosis"] = labels["diagnosis"].astype(int)
    return labels


def plot_class_distribution(labels: pd.DataFrame, output_path: Path) -> None:
    """Save class-count and class-ratio bar plots."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = labels["diagnosis"].value_counts().sort_index()
    ratios = counts / counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar([CLASS_NAMES[idx] for idx in counts.index], counts.values, color="#3B82F6")
    axes[0].set_title("Class counts")
    axes[0].set_ylabel("Images")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar([CLASS_NAMES[idx] for idx in ratios.index], ratios.values, color="#10B981")
    axes[1].set_title("Class ratios")
    axes[1].set_ylabel("Fraction")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_image_size_distribution(quality_report: pd.DataFrame, output_path: Path) -> None:
    """Save image width/height distribution plots."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = quality_report.dropna(subset=["width", "height"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(valid["width"], bins=30, color="#6366F1", alpha=0.85)
    axes[0].set_title("Width distribution")
    axes[0].set_xlabel("Pixels")
    axes[0].set_ylabel("Images")
    axes[1].hist(valid["height"], bins=30, color="#F97316", alpha=0.85)
    axes[1].set_title("Height distribution")
    axes[1].set_xlabel("Pixels")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_samples_per_class(
    labels: pd.DataFrame,
    image_dir: Path,
    output_path: Path,
    image_extension: str,
    samples_per_class: int,
) -> None:
    """Save a grid of raw sample images for every diagnosis class."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_values = sorted(labels["diagnosis"].unique().tolist())
    fig, axes = plt.subplots(
        len(class_values),
        samples_per_class,
        figsize=(3 * samples_per_class, 3 * len(class_values)),
        squeeze=False,
    )

    for row_idx, class_id in enumerate(class_values):
        class_rows = labels[labels["diagnosis"] == class_id].head(samples_per_class)
        for col_idx in range(samples_per_class):
            axis = axes[row_idx][col_idx]
            axis.axis("off")
            if col_idx >= len(class_rows):
                continue
            image_id = str(class_rows.iloc[col_idx]["id_code"])
            image_path = image_dir / f"{image_id}.{image_extension}"
            if image_path.exists():
                with Image.open(image_path) as image:
                    axis.imshow(image.convert("RGB"))
            axis.set_title(f"{CLASS_NAMES[class_id]}\n{image_id}", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def summarize_quality_report(quality_report: pd.DataFrame) -> pd.DataFrame:
    """Aggregate quality flags into a compact summary table."""

    if quality_report.empty:
        return pd.DataFrame(columns=["issue", "count"])

    exploded = (
        quality_report.assign(flags=quality_report["flags"].fillna(""))
        .assign(flags=lambda frame: frame["flags"].str.split(","))
        .explode("flags")
    )
    exploded = exploded[exploded["flags"].astype(bool)]
    summary = exploded["flags"].value_counts().rename_axis("issue").reset_index(name="count")
    duplicate_count = int(quality_report["is_duplicate_phash"].sum())
    if duplicate_count:
        summary = pd.concat(
            [summary, pd.DataFrame([{"issue": "duplicate_or_near_duplicate", "count": duplicate_count}])],
            ignore_index=True,
        )
    return summary


def run_phase1_eda(config: AppConfig) -> dict[str, Path]:
    """Run all Phase 1 EDA tasks and return generated artifact paths."""

    labels = load_train_labels(config.data.train_csv)
    config.eda.figures_dir.mkdir(parents=True, exist_ok=True)
    config.eda.quality_dir.mkdir(parents=True, exist_ok=True)

    thresholds = QualityThresholds(
        black_mean_threshold=config.eda.black_mean_threshold,
        black_std_threshold=config.eda.black_std_threshold,
        low_contrast_std_threshold=config.eda.low_contrast_std_threshold,
        duplicate_hash_size=config.eda.duplicate_hash_size,
    )
    quality_report = build_image_quality_report(
        labels_df=labels,
        image_dir=config.data.train_image_dir,
        thresholds=thresholds,
        image_extension=config.data.image_extension,
    )
    quality_summary = summarize_quality_report(quality_report)

    class_distribution_path = config.eda.figures_dir / "class_distribution.png"
    size_distribution_path = config.eda.figures_dir / "image_size_distribution.png"
    samples_path = config.eda.figures_dir / "samples_per_class.png"
    quality_report_path = config.eda.quality_dir / "image_quality_report.csv"
    quality_summary_path = config.eda.quality_dir / "image_quality_summary.csv"

    plot_class_distribution(labels, class_distribution_path)
    plot_image_size_distribution(quality_report, size_distribution_path)
    plot_samples_per_class(
        labels=labels,
        image_dir=config.data.train_image_dir,
        output_path=samples_path,
        image_extension=config.data.image_extension,
        samples_per_class=config.eda.sample_images_per_class,
    )
    quality_report.to_csv(quality_report_path, index=False)
    quality_summary.to_csv(quality_summary_path, index=False)

    return {
        "class_distribution": class_distribution_path,
        "image_size_distribution": size_distribution_path,
        "samples_per_class": samples_path,
        "quality_report": quality_report_path,
        "quality_summary": quality_summary_path,
    }
