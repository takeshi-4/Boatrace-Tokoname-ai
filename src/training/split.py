"""
Time-based train/validation split utilities.
"""
from __future__ import annotations

import pandas as pd
from typing import Dict, Tuple


def time_split(
    df: pd.DataFrame,
    date_column: str,
    train_end: str,
    val_start: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date to prevent leakage.

    Args:
        df: Input dataframe with a datetime-like column.
        date_column: Column name containing race dates.
        train_end: Inclusive end date for training (YYYY-MM-DD).
        val_start: Inclusive start date for validation (YYYY-MM-DD).

    Returns:
        (train_df, val_df)
    """
    if date_column not in df.columns:
        raise ValueError(f"date column '{date_column}' not found in dataframe")

    dated = df.copy()
    dated[date_column] = pd.to_datetime(dated[date_column], errors="coerce")
    dated = dated.dropna(subset=[date_column])

    train_cutoff = pd.to_datetime(train_end)
    val_cutoff = pd.to_datetime(val_start)

    train_df = dated[dated[date_column] <= train_cutoff].copy()
    val_df = dated[dated[date_column] >= val_cutoff].copy()

    if train_df.empty:
        raise ValueError("time_split produced empty train set; adjust train_end")
    if val_df.empty:
        raise ValueError("time_split produced empty validation set; adjust val_start")

    return train_df, val_df


def load_split_from_config(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg = config.get("data", {})
    split_cfg = data_cfg.get("split", {})
    return time_split(
        df,
        date_column=data_cfg.get("date_column", "date"),
        train_end=split_cfg.get("train_end"),
        val_start=split_cfg.get("val_start"),
    )
