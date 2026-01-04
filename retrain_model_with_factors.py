#!/usr/bin/env python3
"""
Retrain Tokoname P1 prediction model with new factors integrated.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import unicodedata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Add paths for imports
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BOATRACE_PATH = os.path.join(REPO_ROOT, "boatrace-master")
SCRIPTS_PATH = os.path.join(REPO_ROOT, "scripts")
if BOATRACE_PATH not in sys.path:
    sys.path.insert(0, BOATRACE_PATH)
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, SCRIPTS_PATH)

from src.data_preparing.loader import make_race_result_df
from compute_factors import (
    compute_finish_stats,
    compute_racer_weather_stats,
    compute_course_bias,
    compute_weather_suitability_per_racer,
    compute_field_imbalance,
    per_lane_subscores,
    normalize_venue,
    first_nonblank,
)

def extract_factor_score(factor_dict, keyword="score"):
    """Extract numeric score from factor detail string."""
    detail = factor_dict.get("detail", "")
    try:
        import re
        pattern = r"(score|diff)\s+([-+]?\d+\.\d+)"
        match = re.search(pattern, detail)
        if match:
            return float(match.group(2))
    except:
        pass
    # Map state to default score if parsing fails
    state = factor_dict.get("state", "Neutral")
    return {"Positive": 0.5, "Neutral": 0.0, "Negative": -0.5}.get(state, 0.0)

def compute_row_factors(row, finish_stats, racer_weather_stats):
    """Compute factor scores for a single race row."""
    course_factor = compute_course_bias(finish_stats)
    weather_factor = compute_weather_suitability_per_racer(row, racer_weather_stats)
    subs = per_lane_subscores(row)
    field_factor = compute_field_imbalance(subs)
    
    return {
        "course_bias": extract_factor_score(course_factor),
        "weather_suitability": extract_factor_score(weather_factor),
        "field_imbalance": extract_factor_score(field_factor),
    }

def safe_float(x):
    """Convert to float, return None if invalid."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return float(x)
    except:
        return None

def extract_features_from_row(row, finish_stats, racer_weather_stats):
    """Extract all features (old + new) from a race row."""
    # Original features (lane 1 only for P1 prediction)
    motor_place2 = safe_float(first_nonblank(row.get("motor_place2Ratio_1"), row.get("motor_place2Ratio_1_x"), row.get("motor_place2Ratio_1_y")))
    motor_place3 = safe_float(first_nonblank(row.get("motor_place3Ratio_1"), row.get("motor_place3Ratio_1_x"), row.get("motor_place3Ratio_1_y")))
    win_local = safe_float(first_nonblank(row.get("win_rate_local_1"), row.get("win_rate_local_1_x"), row.get("win_rate_local_1_y")))
    win_national = safe_float(first_nonblank(row.get("win_rate_national_1"), row.get("win_rate_national_1_x"), row.get("win_rate_national_1_y")))
    start_time = safe_float(first_nonblank(row.get("ave_start_time_1"), row.get("ave_start_time_1_x"), row.get("ave_start_time_1_y")))
    wind_pow = safe_float(row.get("windPow"))
    wave_height = safe_float(row.get("waveHight"))
    
    # New factors
    factors = compute_row_factors(row, finish_stats, racer_weather_stats)
    
    return {
        "motor_place2Ratio_1": motor_place2,
        "motor_place3Ratio_1": motor_place3,
        "win_rate_local_1": win_local,
        "win_rate_national_1": win_national,
        "ave_start_time_1": start_time,
        "windPow": wind_pow,
        "waveHight": wave_height,
        "course_bias": factors["course_bias"],
        "weather_suitability": factors["weather_suitability"],
        "field_imbalance": factors["field_imbalance"],
    }

def main():
    print("Loading historical race data...")
    df = make_race_result_df()
    
    if df is None or df.empty:
        print("No data loaded; exiting.")
        return
    
    # Filter to Tokoname
    if "venue_clean" not in df.columns and "venue" in df.columns:
        df["venue_clean"] = df["venue"].map(normalize_venue)
    
    target_venue = normalize_venue("常滑")
    venue_col = None
    for col in ["venue_clean", "venue_matched", "venue_x", "venue_y", "venue"]:
        if col in df.columns:
            mask = df[col].map(normalize_venue) == target_venue
            if mask.any():
                df_venue = df[mask].copy()
                print(f"Filtered to {len(df_venue)} races at 常滑 using column '{col}'")
                break
    else:
        print("No venue filter match; using all data")
        df_venue = df.copy()
    
    # Precompute finish and weather stats for factor computation
    print("Computing finish stats and weather stats...")
    finish_stats = compute_finish_stats(df_venue)
    racer_weather_stats = compute_racer_weather_stats(df_venue)
    
    # Extract features and target
    print("Extracting features and targets...")
    features_list = []
    targets = []
    for _, row in df_venue.iterrows():
        # Target: did lane 1 win?
        finish1 = row.get("finish1_lane")
        if pd.isna(finish1):
            continue  # Skip rows without finish data
        
        target = 1 if finish1 == 1 else 0
        
        # Features
        try:
            feats = extract_features_from_row(row, finish_stats, racer_weather_stats)
            # Skip if any critical feature is None
            if any(feats[k] is None for k in ["motor_place2Ratio_1", "win_rate_national_1", "ave_start_time_1"]):
                continue
            features_list.append(feats)
            targets.append(target)
        except Exception as e:
            # Skip problematic rows
            continue
    
    if not features_list:
        print("No valid training examples; exiting.")
        return
    
    # Convert to DataFrame
    X = pd.DataFrame(features_list)
    y = np.array(targets)
    
    print(f"Training on {len(X)} examples ({y.sum()} P1 wins, {len(y)-y.sum()} losses)")
    print(f"Features: {list(X.columns)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train logistic regression
    print("Training LogisticRegression model...")
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\nTest Accuracy: {acc:.3f}")
    print(f"Test ROC-AUC: {auc:.3f}")
    
    # Print feature importances (coefficients)
    print("\nFeature coefficients:")
    for feat, coef in zip(X.columns, model.coef_[0]):
        print(f"  {feat:30s}: {coef:+.4f}")
    
    # Save model and features
    print("\nSaving model and features...")
    joblib.dump(model, "tokoname_p1_model.pkl")
    joblib.dump(list(X.columns), "tokoname_p1_features.pkl")
    print("Saved tokoname_p1_model.pkl and tokoname_p1_features.pkl")
    print("\nModel retraining complete!")

if __name__ == "__main__":
    main()
