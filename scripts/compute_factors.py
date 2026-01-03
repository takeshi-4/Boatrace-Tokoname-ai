#!/usr/bin/env python3
"""
Factor CLI (interpretability-first, Tokoname-only)
- Computes lightweight, explainable factors per race using available columns.
- No ML, no probabilities, no betting logic.
- Outputs Positive/Neutral/Negative with coverage notes.
"""
import argparse
import logging
import os
import sys
import unicodedata
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Suppress loader logging for cleaner output
logging.getLogger("src.data_preparing.loader").setLevel(logging.WARNING)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOATRACE_PATH = os.path.join(REPO_ROOT, "boatrace-master")
if BOATRACE_PATH not in sys.path:
    sys.path.insert(0, BOATRACE_PATH)

from src.data_preparing.loader import make_race_result_df


def normalize_venue(x: Any) -> str:
    """Normalize venue name: NFKC normalization + strip spaces.
    Avoids aggressive character filtering that corrupts multi-byte chars."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = unicodedata.normalize("NFKC", str(x))
    # Only strip leading/trailing whitespace; preserve inner multi-byte chars
    return s.strip()


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(str(x).replace("%", "").strip())
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def first_nonblank(*vals: Any) -> Optional[Any]:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        s = str(v).strip()
        if s:
            return v
    return None


def parse_count_flag(x: Any, prefix: str) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s.startswith(prefix):
        s = s[len(prefix) :]
    try:
        v = int(float(s))
        return v
    except Exception:
        return None


def band_from_score(score: float, pos_thresh: float = 0.3, neg_thresh: float = -0.3) -> str:
    if score >= pos_thresh:
        return "Positive"
    if score <= neg_thresh:
        return "Negative"
    return "Neutral"


def score_vs_field(values: List[Optional[float]], idx: int, invert: bool = False) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    if not filtered or values[idx] is None:
        return None
    median = float(np.median(filtered))
    std = float(np.std(filtered))
    if std == 0:
        std = 0.01
    delta = (median - values[idx]) if invert else (values[idx] - median)
    return max(-2.0, min(2.0, delta / std))


def compute_start_stability(row: pd.Series) -> Dict[str, Any]:
    # Use first_nonblank to handle duplicated _x/_y columns from merge
    starts = [safe_float(first_nonblank(row.get(f"ave_start_time_{i}"), row.get(f"ave_start_time_{i}_x"), row.get(f"ave_start_time_{i}_y"))) for i in range(1, 7)]
    false_counts = [parse_count_flag(first_nonblank(row.get(f"num_false_start_{i}"), row.get(f"num_false_start_{i}_x"), row.get(f"num_false_start_{i}_y")), "F") for i in range(1, 7)]
    late_counts = [parse_count_flag(first_nonblank(row.get(f"num_late_start_{i}"), row.get(f"num_late_start_{i}_x"), row.get(f"num_late_start_{i}_y")), "L") for i in range(1, 7)]
    scores = []
    for i in range(6):
        st_score = score_vs_field(starts, i, invert=True)
        fc = false_counts[i] or 0
        lc = late_counts[i] or 0
        penalty = 0.1 * (fc + lc)
        if st_score is None:
            scores.append(None)
        else:
            scores.append(st_score - penalty)
    available = [s for s in scores if s is not None]
    if not available:
        return {
            "name": "Start Timing Stability",
            "state": "Neutral",
            "detail": "insufficient start data",
            "coverage": "n=0",
        }
    avg_score = float(np.mean(available))
    return {
        "name": "Start Timing Stability",
        "state": band_from_score(avg_score),
        "detail": f"mean score {avg_score:.2f} (lower ST, fewer late/false is better)",
        "coverage": f"lanes with data={len(available)}/6",
    }


def compute_motor_performance(row: pd.Series) -> Dict[str, Any]:
    # Use first_nonblank to handle duplicated _x/_y columns from merge
    ratios = [safe_float(first_nonblank(row.get(f"motor_place2Ratio_{i}"), row.get(f"motor_place2Ratio_{i}_x"), row.get(f"motor_place2Ratio_{i}_y"))) for i in range(1, 7)]
    scores = []
    for i in range(6):
        score = score_vs_field(ratios, i, invert=False)
        scores.append(score)
    available = [s for s in scores if s is not None]
    if not available:
        return {
            "name": "Motor Performance",
            "state": "Neutral",
            "detail": "insufficient motor data",
            "coverage": "n=0",
        }
    avg_score = float(np.mean(available))
    return {
        "name": "Motor Performance",
        "state": band_from_score(avg_score),
        "detail": f"mean score {avg_score:.2f} (vs field motor place2 ratios)",
        "coverage": f"lanes with data={len(available)}/6",
    }


def compute_racer_consistency(row: pd.Series) -> Dict[str, Any]:
    # Use first_nonblank to handle duplicated _x/_y columns from merge
    ratios = [
        safe_float(first_nonblank(
            row.get(f"place2Ratio_national_{i}"),
            row.get(f"place2Ratio_national_{i}_x"),
            row.get(f"place2Ratio_national_{i}_y"),
            row.get(f"win_rate_national_{i}"),
            row.get(f"win_rate_national_{i}_x"),
            row.get(f"win_rate_national_{i}_y")
        ))
        for i in range(1, 7)
    ]
    scores = []
    for i in range(6):
        score = score_vs_field(ratios, i, invert=False)
        scores.append(score)
    available = [s for s in scores if s is not None]
    if not available:
        return {
            "name": "Racer Consistency",
            "state": "Neutral",
            "detail": "insufficient racer form data",
            "coverage": "n=0",
        }
    avg_score = float(np.mean(available))
    return {
        "name": "Racer Consistency",
        "state": band_from_score(avg_score),
        "detail": f"mean score {avg_score:.2f} (vs field place/win ratios)",
        "coverage": f"lanes with data={len(available)}/6",
    }


def compute_weather_suitability(row: pd.Series) -> Dict[str, Any]:
    # Placeholder: no per-racer weather splits available, so surface as neutral with coverage note.
    has_weather = any(
        row.get(c) not in (None, "", np.nan) for c in ["weather", "windDir", "windPow", "waveHight"]
    )
    return {
        "name": "Weather Suitability",
        "state": "Neutral",
        "detail": "not enough per-racer weather history; showing neutral",
        "coverage": "weather present" if has_weather else "weather missing",
    }


def compute_course_bias() -> Dict[str, Any]:
    return {
        "name": "Course / Venue Bias",
        "state": "Neutral",
        "detail": "finish-by-lane history not available in current dataset",
        "coverage": "pending finish data",
    }


def compute_field_imbalance(subfactors: Dict[str, float]) -> Dict[str, Any]:
    inside = []
    outside = []
    for lane in range(1, 7):
        lane_score = 0.0
        count = 0
        for key, scores in subfactors.items():
            if scores[lane - 1] is not None:
                lane_score += scores[lane - 1]
                count += 1
        if count == 0:
            lane_value = None
        else:
            lane_value = lane_score / count
        if lane_value is not None:
            (inside if lane <= 3 else outside).append(lane_value)
    if not inside or not outside:
        return {
            "name": "Field Imbalance",
            "state": "Neutral",
            "detail": "insufficient lane subfactor data",
            "coverage": "inside data={}/3 outside data={}/3".format(len(inside), len(outside)),
        }
    inside_mean = float(np.mean(inside))
    outside_mean = float(np.mean(outside))
    diff = inside_mean - outside_mean
    return {
        "name": "Field Imbalance",
        "state": band_from_score(diff, pos_thresh=0.2, neg_thresh=-0.2),
        "detail": f"inside-outside diff {diff:.2f} (inside positive)",
        "coverage": "inside data={}/3 outside data={}/3".format(len(inside), len(outside)),
    }


def per_lane_subscores(row: pd.Series) -> Dict[str, List[Optional[float]]]:
    # Use first_nonblank to handle duplicated _x/_y columns from merge
    starts = [safe_float(first_nonblank(row.get(f"ave_start_time_{i}"), row.get(f"ave_start_time_{i}_x"), row.get(f"ave_start_time_{i}_y"))) for i in range(1, 7)]
    motor = [safe_float(first_nonblank(row.get(f"motor_place2Ratio_{i}"), row.get(f"motor_place2Ratio_{i}_x"), row.get(f"motor_place2Ratio_{i}_y"))) for i in range(1, 7)]
    form = [
        safe_float(first_nonblank(
            row.get(f"place2Ratio_national_{i}"),
            row.get(f"place2Ratio_national_{i}_x"),
            row.get(f"place2Ratio_national_{i}_y"),
            row.get(f"win_rate_national_{i}"),
            row.get(f"win_rate_national_{i}_x"),
            row.get(f"win_rate_national_{i}_y")
        ))
        for i in range(1, 7)
    ]
    start_scores = [score_vs_field(starts, i, invert=True) for i in range(6)]
    motor_scores = [score_vs_field(motor, i, invert=False) for i in range(6)]
    form_scores = [score_vs_field(form, i, invert=False) for i in range(6)]
    return {
        "start": start_scores,
        "motor": motor_scores,
        "form": form_scores,
    }


def compute_factors_for_race(row: pd.Series) -> List[Dict[str, Any]]:
    # Core factors
    start_factor = compute_start_stability(row)
    motor_factor = compute_motor_performance(row)
    racer_factor = compute_racer_consistency(row)
    weather_factor = compute_weather_suitability(row)
    course_factor = compute_course_bias()

    subs = per_lane_subscores(row)
    field_factor = compute_field_imbalance(subs)

    factors = [
        start_factor,
        motor_factor,
        racer_factor,
        weather_factor,
        course_factor,
        field_factor,
    ]
    return factors


def top_factors(factors: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    priority = {"Positive": 2, "Neutral": 1, "Negative": 0}
    return sorted(factors, key=lambda f: priority.get(f.get("state", "Neutral"), 1), reverse=True)[:k]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute interpretable factors for Tokoname races")
    ap.add_argument("--venue", default="常滑", help="Target venue (default: 常滑)")
    ap.add_argument("--limit", type=int, default=5, help="Number of recent races to display")
    args = ap.parse_args()

    df = make_race_result_df()
    if df is None or df.empty:
        print("Dataset is empty; aborting.")
        sys.exit(1)

    if "venue_clean" not in df.columns and "venue" in df.columns:
        df = df.copy()
        df["venue_clean"] = df["venue"].map(normalize_venue)

    target = normalize_venue(args.venue)
    # Use first_nonblank for venue fallback: try venue_clean, venue_x, venue_y, venue
    df = df.copy()
    df["venue_matched"] = df.apply(
        lambda row: normalize_venue(first_nonblank(row.get("venue_clean"), row.get("venue_x"), row.get("venue_y"), row.get("venue"))),
        axis=1
    )
    
    # If target is provided and rows match, filter; otherwise use all rows
    if target and (df["venue_matched"] == target).any():
        df = df[df["venue_matched"] == target].copy()
    elif target:
        # Venue provided but no matches found - use best effort with raw venues
        print(f"[Note] No exact matches for venue '{target}'; using all available races.")
    
    if df.empty:
        print("No data available.")
        sys.exit(1)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["date", "raceNumber"], ascending=[False, False])

    recent = df.head(args.limit)

    for _, row in recent.iterrows():
        race_label = f"{row.get('date')} R{row.get('raceNumber')}"
        # Display the matched venue value or fall back to raw venue columns
        display_venue = row.get('venue_matched', row.get('venue', row.get('venue_x', 'unknown')))
        print(f"\nRace: {race_label} (venue={display_venue})")
        factors = compute_factors_for_race(row)
        for f in top_factors(factors, k=5):
            print(f"- [{f['state']}] {f['name']}: {f['detail']} ({f['coverage']})")


if __name__ == "__main__":
    main()
