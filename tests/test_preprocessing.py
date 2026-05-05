import numpy as np

from dr_grading.data.preprocessing import (
    PreprocessOptions,
    ben_graham_preprocess,
    circle_crop,
    crop_black_border,
    preprocess_fundus_image,
    resize_keep_aspect,
)


def test_crop_black_border_removes_dark_padding() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 4:16] = 120

    cropped = crop_black_border(image)

    assert cropped.shape == (10, 12, 3)


def test_preprocess_fundus_image_returns_square_uint8() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[10:70, 20:80] = [80, 120, 90]
    options = PreprocessOptions(image_size=64)

    processed = preprocess_fundus_image(image, options=options, method="ben_graham")

    assert processed.shape == (64, 64, 3)
    assert processed.dtype == np.uint8


def test_circle_crop_masks_corners() -> None:
    image = np.full((64, 64, 3), 255, dtype=np.uint8)

    cropped = circle_crop(image)

    assert cropped[0, 0].sum() == 0
    assert cropped[32, 32].sum() == 255 * 3


def test_resize_keep_aspect_preserves_square_shape() -> None:
    image = np.full((20, 40, 3), 100, dtype=np.uint8)

    resized = resize_keep_aspect(image, 64)

    assert resized.shape == (64, 64, 3)


def test_ben_graham_preprocess_preserves_shape() -> None:
    image = np.full((32, 32, 3), 128, dtype=np.uint8)

    processed = ben_graham_preprocess(image)

    assert processed.shape == image.shape
    assert processed.dtype == np.uint8
