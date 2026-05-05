from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("torch")

from dr_grading.data.dataset import RetinopathyDataset, compute_class_weights
from dr_grading.data.folds import create_stratified_folds


def test_create_stratified_folds_assigns_all_rows() -> None:
    labels = pd.DataFrame(
        {
            "id_code": [f"img_{idx}" for idx in range(20)],
            "diagnosis": [0, 1, 2, 3, 4] * 4,
        }
    )

    folded = create_stratified_folds(labels, n_splits=2, seed=42)

    assert set(folded["fold"].unique()) == {0, 1}
    assert (folded["fold"] >= 0).all()


def test_compute_class_weights_returns_one_weight_per_class() -> None:
    labels = pd.Series([0, 0, 0, 1, 2, 3, 4])

    weights = compute_class_weights(labels, num_classes=5)

    assert weights.shape[0] == 5
    assert weights[0] < weights[4]


def test_dataset_reads_image_and_label(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    Image.fromarray(image).save(image_dir / "sample.png")
    frame = pd.DataFrame([{"id_code": "sample", "diagnosis": 2}])
    dataset = RetinopathyDataset(frame, image_dir=image_dir, task="classification")

    item = dataset[0]

    assert item["image"].shape == (3, 16, 16)
    assert int(item["target"]) == 2
