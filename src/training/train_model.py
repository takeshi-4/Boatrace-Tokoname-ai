"""
Baseline training utilities for Tokoname.
"""
from __future__ import annotations

import os
import re
import json
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional dependency
    lgb = None


def build_factor_map(factor_config: Dict) -> Dict[str, str]:
    """Map feature name to factor name using factors_config.yaml."""
    factor_map: Dict[str, str] = {}
    for factor_name, cfg in factor_config.items():
        if not isinstance(cfg, dict):
            continue
        for col in cfg.get("raw_columns", []) or []:
            factor_map[col] = factor_name
        for feat in cfg.get("derived_features", []) or []:
            fname = feat.get("name")
            if fname:
                factor_map[f"derived_{fname}"] = factor_name
    return factor_map


def _strip_lane_suffix(col: str) -> Tuple[str, str]:
    match = re.match(r"(.+)_([1-6])$", col)
    if match:
        return match.group(1), match.group(2)
    return col, ""


def expand_lane_samples(
    df: pd.DataFrame,
    target_column: str = "finish1_lane",
    id_columns: List[str] | None = None,
) -> pd.DataFrame:
    """Expand wide race rows into per-lane samples with binary label."""
    if id_columns is None:
        id_columns = ["date", "raceNumber", "venue_clean"]

    if target_column not in df.columns:
        raise ValueError(f"target column '{target_column}' not found")

    target = pd.to_numeric(df[target_column], errors="coerce")
    id_cols = [c for c in id_columns if c in df.columns]

    wide_cols = [c for c in df.columns if c not in id_cols]
    exclude_cols = {target_column, "finish2_lane", "finish3_lane", "placeBed"}

    samples = []
    for lane in range(1, 7):
        lane_rows = df[id_cols].copy()
        lane_rows["lane"] = lane
        lane_rows["label"] = (target == lane).astype(int)

        lane_features = {}
        for col in wide_cols:
            if col in exclude_cols:
                continue
            base, lane_suffix = _strip_lane_suffix(col)
            if lane_suffix and lane_suffix != str(lane):
                continue  # lane-specific but not this lane
            if lane_suffix:
                feature_name = base
            else:
                feature_name = col
            lane_features[feature_name] = df[col].values

        lane_df = lane_rows.join(pd.DataFrame(lane_features))
        samples.append(lane_df)

    out = pd.concat(samples, axis=0, ignore_index=True)
    out = out.dropna(subset=["label"])
    return out


def prepare_features(
    df: pd.DataFrame,
    feature_map: Dict[str, str] | None = None,
    drop_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Select and numeric-cast features, dropping identifiers and label."""
    if drop_columns is None:
        drop_columns = ["label", "lane", "raceNumber", "date", "venue_clean"]

    feature_cols = [c for c in df.columns if c not in drop_columns]
    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)
    return X, feature_cols


def train_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    model_cfg: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """Train LightGBM baseline; fallback to GradientBoosting if unavailable."""
    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_val = val_df[feature_cols]
    y_val = val_df["label"]

    model_type = (model_cfg or {}).get("type", "lightgbm")
    params = (model_cfg or {}).get("params", {})

    if model_type == "lightgbm" and lgb is not None:
        model = lgb.LGBMClassifier(**params)
    else:
        model = GradientBoostingClassifier(random_state=params.get("random_state", 42))

    model.fit(X_train, y_train)
    val_pred = model.predict_proba(X_val)[:, 1]
    metrics = {"val_logloss": float(log_loss(y_val, val_pred))}

    return model, metrics


def save_artifacts(
    output_dir: str,
    model: Any,
    feature_cols: List[str],
    factor_map: Dict[str, str],
):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "model.pkl"))
    joblib.dump(feature_cols, os.path.join(output_dir, "feature_cols.pkl"))
    with open(os.path.join(output_dir, "factor_map.json"), "w", encoding="utf-8") as f:
        json.dump(factor_map, f, ensure_ascii=False, indent=2)
