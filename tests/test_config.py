from pathlib import Path

from dr_grading.config import load_config


def test_load_config_defaults() -> None:
    config = load_config(Path("configs/config.yaml"))

    assert config.data.num_classes == 5
    assert config.training.folds == 5
    assert config.model.task in {"classification", "regression", "ordinal"}
