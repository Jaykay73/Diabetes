"""Albumentations transform factories for DR grading."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


APTOS_MEAN = (0.485, 0.456, 0.406)
APTOS_STD = (0.229, 0.224, 0.225)


def build_train_transforms(image_size: int) -> A.Compose:
    """Training augmentations used for robust fundus grading."""

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=25,
                border_mode=0,
                value=0,
                p=0.65,
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=max(8, image_size // 16),
                max_width=max(8, image_size // 16),
                min_holes=1,
                fill_value=0,
                p=0.25,
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.20, border_mode=0, value=0, p=0.20),
            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=12,
                val_shift_limit=8,
                p=0.35,
            ),
            A.Normalize(mean=APTOS_MEAN, std=APTOS_STD),
            ToTensorV2(),
        ]
    )


def build_valid_transforms(image_size: int) -> A.Compose:
    """Validation/test transforms: resize and normalize only."""

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=APTOS_MEAN, std=APTOS_STD),
            ToTensorV2(),
        ]
    )
