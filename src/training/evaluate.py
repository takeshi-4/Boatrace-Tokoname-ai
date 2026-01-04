"""
Evaluation utilities: AUC, LogLoss, Top-K hit rate, calibration bins.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss


def top_k_hit_rate(
    df: pd.DataFrame,
    prob_column: str,
    race_id_cols: List[str],
    top_k: int = 3,
) -> float:
    """Compute per-race top-k hit rate."""
    grouped = df.groupby(race_id_cols)
    hits = []
    for _, group in grouped:
        top = group.sort_values(prob_column, ascending=False).head(top_k)
        hits.append(int(top["label"].max() == 1))
    return float(np.mean(hits)) if hits else 0.0


def calibration_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> List[Dict]:
    bins = []
    prob_sorted = np.sort(probs)
    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (probs >= edges[i]) & (probs < edges[i + 1]) if i < n_bins - 1 else (probs >= edges[i])
        if mask.sum() == 0:
            bins.append({"bin": i, "count": 0, "pred": None, "emp": None})
            continue
        bins.append({
            "bin": i,
            "count": int(mask.sum()),
            "pred": float(probs[mask].mean()),
            "emp": float(labels[mask].mean()),
        })
    return bins


def evaluate_predictions(
    df: pd.DataFrame,
    prob_column: str,
    race_id_cols: List[str],
    top_k: int = 3,
    n_bins: int = 10,
) -> Dict[str, float]:
    y_true = df["label"].values
    y_prob = df[prob_column].values
    metrics = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob)),
        "top_k_hit_rate": top_k_hit_rate(df, prob_column, race_id_cols, top_k=top_k),
    }
    metrics["calibration_bins"] = calibration_bins(y_prob, y_true, n_bins=n_bins)
    return metrics
