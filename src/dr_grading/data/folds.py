"""Stratified fold creation for diabetic retinopathy grading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_stratified_folds(
    labels: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    target_col: str = "diagnosis",
) -> pd.DataFrame:
    """Assign a fold id using Stratified K-Fold, preserving DR class ratios."""

    required = {"id_code", target_col}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Labels are missing required columns: {sorted(missing)}")

    folded = labels.copy()
    folded["fold"] = -1
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (_, valid_idx) in enumerate(splitter.split(folded["id_code"], folded[target_col])):
        folded.loc[valid_idx, "fold"] = fold

    folded["fold"] = folded["fold"].astype(int)
    return folded


def save_stratified_folds(
    train_csv: Path,
    output_csv: Path,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Read train.csv, create stratified folds, and save the fold CSV."""

    labels = pd.read_csv(train_csv)
    folded = create_stratified_folds(labels=labels, n_splits=n_splits, seed=seed)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    folded.to_csv(output_csv, index=False)
    return folded


def split_fold(folded: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train and validation frames for one fold."""

    if "fold" not in folded.columns:
        raise ValueError("Folded dataframe must contain a fold column")
    train_df = folded[folded["fold"] != fold].reset_index(drop=True)
    valid_df = folded[folded["fold"] == fold].reset_index(drop=True)
    return train_df, valid_df
