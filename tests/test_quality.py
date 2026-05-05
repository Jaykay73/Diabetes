import numpy as np

from dr_grading.data.quality import QualityThresholds, flag_quality_issues, image_stats


def test_black_image_is_flagged() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    stats = image_stats(image)
    thresholds = QualityThresholds(
        black_mean_threshold=8.0,
        black_std_threshold=5.0,
        low_contrast_std_threshold=18.0,
        duplicate_hash_size=16,
    )

    flags = flag_quality_issues(stats, thresholds)

    assert "black_or_nearly_black" in flags
    assert "low_contrast" in flags


def test_high_contrast_image_not_low_contrast() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, 32:] = 255
    stats = image_stats(image)
    thresholds = QualityThresholds(
        black_mean_threshold=8.0,
        black_std_threshold=5.0,
        low_contrast_std_threshold=18.0,
        duplicate_hash_size=16,
    )

    flags = flag_quality_issues(stats, thresholds)

    assert "low_contrast" not in flags
