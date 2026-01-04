import os
import sys
import argparse
import unicodedata
import yaml
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "boatrace-master"))

from src.data_preparing.loader import make_race_result_df
from src.factors.factor_binder import FactorBinder
from src.training.split import load_split_from_config
from src.training.train_model import (
    build_factor_map,
    expand_lane_samples,
    prepare_features,
    train_baseline,
    save_artifacts,
)
from src.training.evaluate import evaluate_predictions
from src.explain.factor_explain import explain_samples


def normalize_venue(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = unicodedata.normalize("NFKC", str(x))
    return "".join(ch for ch in s if not ch.isspace()).strip()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_merged_race_df():
    txt_glob = os.path.join(
        BASE_DIR, "boatrace-master", "data", "results_race", "K1*.TXT"
    )
    print("[TXT GLOB]", txt_glob)

    df = make_race_result_df(race_results_file_path=txt_glob)

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    if "venue_clean" not in df.columns and "venue" in df.columns:
        df["venue_clean"] = df["venue"].map(normalize_venue)

    return df


def main():
    ap = argparse.ArgumentParser(description="Phase 3: train + evaluate + explain")
    ap.add_argument("--venue", default="常滑", help="Target venue")
    ap.add_argument("--config", default=os.path.join("configs", "train.yaml"))
    args = ap.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    race_id_cols = data_cfg.get("race_id_columns", ["date", "raceNumber", "venue_clean"])
    top_k = data_cfg.get("top_k", 3)

    df = load_merged_race_df()
    if df.empty:
        print("[STOP] merged dataframe is empty.")
        return

    target_clean = normalize_venue(args.venue or data_cfg.get("venue", "常滑"))
    df = df[df["venue_clean"] == target_clean]
    if df.empty:
        print(f"[STOP] no rows for venue {target_clean}")
        return

    print(f"[INFO] Loaded {len(df)} rows for {target_clean}")

    binder = FactorBinder(os.path.join(BASE_DIR, "factors_config.yaml"))
    bind_result = binder.bind(df)
    factor_df = bind_result["factor_df"]
    if bind_result.get("warnings"):
        print("[WARN]", bind_result["warnings"])

    factor_config = binder.config
    factor_map = build_factor_map(factor_config)

    samples = expand_lane_samples(
        factor_df,
        target_column=data_cfg.get("target_column", "finish1_lane"),
        id_columns=race_id_cols,
    )
    samples = samples.sort_values(by=[data_cfg.get("date_column", "date"), "raceNumber", "lane"])

    train_df, val_df = load_split_from_config(samples, config)
    print(f"[INFO] Split: train={len(train_df)} rows, val={len(val_df)} rows")

    X_train, feature_cols = prepare_features(train_df)
    X_val = val_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    train_df[feature_cols] = X_train
    val_df[feature_cols] = X_val

    model, train_metrics = train_baseline(
        train_df,
        val_df,
        feature_cols,
        model_cfg=config.get("model", {}),
    )

    val_df["pred_prob"] = model.predict_proba(val_df[feature_cols])[:, 1]
    metrics = evaluate_predictions(
        val_df,
        prob_column="pred_prob",
        race_id_cols=race_id_cols,
        top_k=top_k,
        n_bins=config.get("evaluate", {}).get("calibration_bins", 10),
    )

    artifacts_dir = os.path.join(BASE_DIR, "artifacts", "phase3_tokoname")
    save_artifacts(artifacts_dir, model, feature_cols, factor_map)
    print(f"[INFO] Artifacts saved to {artifacts_dir}")

    print("[METRICS]", {k: v for k, v in metrics.items() if k != "calibration_bins"})

    sample_size = config.get("explain", {}).get("sample_size", 5)
    top_features = config.get("explain", {}).get("top_features", 5)
    sample_X = val_df.head(sample_size)[feature_cols]
    explanations = explain_samples(model, sample_X, factor_map, top_features=top_features)
    print("[EXPLAIN] sample", explanations[:3])


if __name__ == "__main__":
    main()
