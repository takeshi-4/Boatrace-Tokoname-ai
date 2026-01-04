import os
import sys
import joblib
import numpy as np
import pandas as pd

# Add boatrace-master to path for loader and factor computation
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BOATRACE_PATH = os.path.join(REPO_ROOT, "boatrace-master")
if BOATRACE_PATH not in sys.path:
    sys.path.insert(0, BOATRACE_PATH)

# Import factor computation functions from scripts
SCRIPTS_PATH = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, SCRIPTS_PATH)

from compute_factors import (
    compute_finish_stats,
    compute_racer_weather_stats,
    compute_course_bias,
    compute_weather_suitability_per_racer,
    compute_field_imbalance,
    per_lane_subscores,
)

# 学習済みモデルと特徴量リストを読み込み
model = joblib.load("tokoname_p1_model.pkl")
feature_cols = joblib.load("tokoname_p1_features.pkl")

def compute_new_factors(row: pd.Series, finish_stats: dict, racer_weather_stats: dict) -> dict:
    """Compute new factor scores and extract numeric values."""
    # Compute factors
    course_factor = compute_course_bias(finish_stats)
    weather_factor = compute_weather_suitability_per_racer(row, racer_weather_stats)
    
    subs = per_lane_subscores(row)
    field_factor = compute_field_imbalance(subs)
    
    # Extract numeric scores from factor dictionaries
    # Parse detail strings to get scores (e.g., "inside-outside top3 rate diff 0.25" -> 0.25)
    def extract_score(factor_dict: dict, keyword: str = "score") -> float:
        """Extract numeric score from factor detail string."""
        detail = factor_dict.get("detail", "")
        try:
            # Look for patterns like "score X.XX" or "diff X.XX"
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
    
    course_bias_score = extract_score(course_factor)
    weather_suitability_score = extract_score(weather_factor)
    field_imbalance_score = extract_score(field_factor)
    
    return {
        "course_bias": course_bias_score,
        "weather_suitability": weather_suitability_score,
        "field_imbalance": field_imbalance_score,
    }


def predict_p1_win_prob(features: dict, venue_history_df: pd.DataFrame = None) -> float:
    """
    features: 1レース分の特徴量を dict で渡す
      例:
      {
        "motor_place2Ratio_1": 52.3,
        "motor_place3Ratio_1": 68.1,
        "win_rate_local_1": 35.0,
        "win_rate_national_1": 28.5,
        "ave_start_time_1": 0.15,
        "windPow": 4,
        "waveHight": 3,
      }
    venue_history_df: Optional DataFrame for computing contextual factors
    """
    # If venue history is provided, compute new factors
    if venue_history_df is not None and not venue_history_df.empty:
        # Create a row Series from features dict for factor computation
        row = pd.Series(features)
        
        # Compute historical stats needed for factors
        finish_stats = compute_finish_stats(venue_history_df)
        racer_weather_stats = compute_racer_weather_stats(venue_history_df)
        
        # Compute new factors
        new_factors = compute_new_factors(row, finish_stats, racer_weather_stats)
        
        # Merge new factors into features dict
        features = {**features, **new_factors}
    
    # DataFrame にしてモデルに突っ込む
    df = pd.DataFrame([features], columns=feature_cols)
    prob = model.predict_proba(df)[0, 1]
    return float(prob)

if __name__ == "__main__":
    from src.data_preparing.loader import make_race_result_df
    
    # Load historical data for Tokoname to compute contextual factors
    print("Loading historical race data for factor computation...")
    df = make_race_result_df()
    
    # Filter to Tokoname venue
    if df is not None and not df.empty:
        # Normalize venue names
        import unicodedata
        def normalize_venue(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return ""
            s = unicodedata.normalize("NFKC", str(x))
            return s.strip()
        
        if "venue_clean" not in df.columns and "venue" in df.columns:
            df["venue_clean"] = df["venue"].map(normalize_venue)
        
        target_venue = normalize_venue("常滑")
        # Try multiple venue columns
        venue_col = None
        for col in ["venue_clean", "venue_matched", "venue_x", "venue_y", "venue"]:
            if col in df.columns:
                mask = df[col].map(normalize_venue) == target_venue
                if mask.any():
                    venue_history = df[mask].copy()
                    print(f"Found {len(venue_history)} historical races for 常滑")
                    break
        else:
            venue_history = None
            print("Warning: No venue history found; using basic features only")
    else:
        venue_history = None
        print("Warning: Could not load data; using basic features only")
    
    # テスト用に仮のレース条件を入れてみる
    sample_features = {
        "motor_place2Ratio_1": 52.3,
        "motor_place3Ratio_1": 68.1,
        "win_rate_local_1": 35.0,
        "win_rate_national_1": 28.5,
        "ave_start_time_1": 0.15,
        "windPow": 4,
        "waveHight": 3,
    }

    prob = predict_p1_win_prob(sample_features, venue_history)
    print(f"このレースで1号艇が勝つ確率: {prob:.3f}（{prob*100:.1f}%）")
