"""
Factor-level explanation utilities using SHAP (if available) or gain importances.
"""
from __future__ import annotations

import json
from typing import Dict, List, Any

import numpy as np
import pandas as pd

try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:  # pragma: no cover - optional dependency
    shap = None
    _HAS_SHAP = False


def factor_importance_from_gain(model, factor_map: Dict[str, str], feature_cols: List[str]) -> Dict[str, float]:
    if not hasattr(model, "feature_importances_"):
        return {}
    gains = model.feature_importances_
    importance = {}
    for feat, gain in zip(feature_cols, gains):
        factor = factor_map.get(feat, "unknown")
        importance[factor] = importance.get(factor, 0.0) + float(gain)
    return importance


def explain_samples(
    model: Any,
    X: pd.DataFrame,
    factor_map: Dict[str, str],
    top_features: int = 5,
) -> List[Dict[str, Any]]:
    """Return per-sample factor contributions (SHAP if available)."""
    results = []
    use_shap = _HAS_SHAP and hasattr(model, "predict_proba")

    if use_shap:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # Binary classifier: shap_values is list [neg, pos]; take pos
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        shap_array = np.array(shap_values)
        for i in range(min(len(X), shap_array.shape[0])):
            row_shap = shap_array[i]
            contrib = {}
            feature_breakdown = []
            for feat, val in zip(X.columns, row_shap):
                factor = factor_map.get(feat, "unknown")
                contrib[factor] = contrib.get(factor, 0.0) + float(val)
                feature_breakdown.append((feat, factor, float(val)))
            sorted_factors = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)
            top_feat = sorted(feature_breakdown, key=lambda x: abs(x[2]), reverse=True)[:top_features]
            prob = float(model.predict_proba(X.iloc[[i]])[:, 1][0])
            results.append({
                "index": int(i),
                "pred_prob": prob,
                "factor_contributions": sorted_factors,
                "top_features": top_feat,
            })
        return results

    # Fallback: use gain importances (no per-sample detail)
    gains = factor_importance_from_gain(model, factor_map, list(X.columns))
    sorted_factors = sorted(gains.items(), key=lambda x: x[1], reverse=True)
    results.append({
        "pred_prob": float(model.predict_proba(X.head(1))[:, 1][0]) if hasattr(model, "predict_proba") else None,
        "factor_contributions": sorted_factors,
        "note": "SHAP not available; using global gain importances",
    })
    return results
