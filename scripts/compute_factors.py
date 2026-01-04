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

# Approximate course bearings (degrees) per venue; tweak as better info becomes available
COURSE_BEARINGS: Dict[str, float] = {
    "常滑": 72.5,  # Tokoname: 70–75° (ENE–WSW)
    "大村": 107.5,  # Omura: 105–110° (E by SE–WNW)
}


def normalize_venue(x: Any) -> str:
    """Normalize venue name: NFKC normalization + strip spaces.
    Avoids aggressive character filtering that corrupts multi-byte chars."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = unicodedata.normalize("NFKC", str(x))
    # Only strip leading/trailing whitespace; preserve inner multi-byte chars
    return s.strip()


def wind_dir_to_deg(x: Any) -> Optional[float]:
    """Map Japanese/compass wind direction strings to degrees (0=N)."""
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None

    # Standard 16-point compass (Japanese)
    dir_map = {
        "北": 0.0,
        "北北東": 22.5,
        "北東": 45.0,
        "東北東": 67.5,
        "東": 90.0,
        "東南東": 112.5,
        "南東": 135.0,
        "南南東": 157.5,
        "南": 180.0,
        "南南西": 202.5,
        "南西": 225.0,
        "西南西": 247.5,
        "西": 270.0,
        "西北西": 292.5,
        "北西": 315.0,
        "北北西": 337.5,
    }

    # Fall back to simple cardinal parsing (English letters)
    simple_map = {
        "N": 0.0,
        "NE": 45.0,
        "E": 90.0,
        "SE": 135.0,
        "S": 180.0,
        "SW": 225.0,
        "W": 270.0,
        "NW": 315.0,
    }

    if s in dir_map:
        return dir_map[s]
    su = s.upper()
    if su in simple_map:
        return simple_map[su]
    return None


def _course_bearing(row: pd.Series) -> float:
    venue_raw = first_nonblank(
        row.get("venue_matched"),
        row.get("venue_clean"),
        row.get("venue_x"),
        row.get("venue_y"),
        row.get("venue"),
    )
    venue_norm = normalize_venue(venue_raw)
    return COURSE_BEARINGS.get(venue_norm, 0.0)


def _wind_sector(row: pd.Series) -> Optional[str]:
    """Return relative wind sector (head/tail/cross-*) vs course bearing."""
    dir_raw = first_nonblank(row.get("windDir"), row.get("windDir_x"), row.get("windDir_y"))
    wind_deg = wind_dir_to_deg(dir_raw)
    if wind_deg is None:
        return None

    rel = (wind_deg - _course_bearing(row)) % 360
    if rel < 45 or rel >= 315:
        return "head"
    if 135 <= rel < 225:
        return "tail"
    if 45 <= rel < 135:
        return "cross-starboard"
    return "cross-port"


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(str(x).replace("%", "").strip())
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    """Convert to int safely. Returns None for empty/NaN/invalid."""
    if x is None:
        return None
    try:
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        return int(float(s))
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


def band_from_score(score: float, pos_thresh: float = 0.1, neg_thresh: float = -0.1) -> str:
    """Map a continuous score to a sentiment band with low neutrality gap."""
    if score >= pos_thresh:
        return "Positive"
    if score <= neg_thresh:
        return "Negative"
    return "Neutral"


def strength_from_score(score: float) -> str:
    """Map absolute magnitude to strength descriptor (Phase 3)."""
    abs_score = abs(score)
    if abs_score >= 0.5:
        return "strong"
    elif abs_score >= 0.2:
        return "moderate"
    elif abs_score >= 0.1:
        return "weak"
    else:
        return "minimal"


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
    strength = strength_from_score(avg_score)
    return {
        "name": "Start Timing Stability",
        "state": band_from_score(avg_score),
        "score": avg_score,
        "strength": strength,
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
    strength = strength_from_score(avg_score)
    return {
        "name": "Motor Performance",
        "state": band_from_score(avg_score),
        "score": avg_score,
        "strength": strength,
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
    strength = strength_from_score(avg_score)
    return {
        "name": "Racer Consistency",
        "state": band_from_score(avg_score),
        "score": avg_score,
        "strength": strength,
        "detail": f"mean score {avg_score:.2f} (vs field place/win ratios)",
        "coverage": f"lanes with data={len(available)}/6",
    }


def compute_weather_suitability(row: pd.Series) -> Dict[str, Any]:
    raise NotImplementedError  # replaced by compute_weather_suitability_per_racer


def compute_finish_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate finish-by-lane history for the current venue subset."""
    if df.empty or "finish1_lane" not in df.columns:
        return {"top1": [None] * 6, "top3": [None] * 6, "total": 0}

    mask = df["finish1_lane"].notna()
    total = int(mask.sum())
    if total == 0:
        return {"top1": [None] * 6, "top3": [None] * 6, "total": 0}

    top1_rates = []
    top3_rates = []
    for lane in range(1, 7):
        top1 = float((df["finish1_lane"] == lane).sum()) / total
        top3 = float(
            ((df["finish1_lane"] == lane)
             | (df.get("finish2_lane") == lane)
             | (df.get("finish3_lane") == lane)).sum()
        ) / total
        top1_rates.append(top1)
        top3_rates.append(top3)

    return {"top1": top1_rates, "top3": top3_rates, "total": total}


def compute_course_bias(finish_stats: Dict[str, Any]) -> Dict[str, Any]:
    if not finish_stats or finish_stats.get("total", 0) == 0:
        return {
            "name": "Course / Venue Bias",
            "state": "Neutral",
            "detail": "finish-by-lane history unavailable",
            "coverage": "pending finish data",
        }

    inside = finish_stats["top3"][0:3]
    outside = finish_stats["top3"][3:6]
    inside_rate = float(np.mean(inside)) if inside else 0.0
    outside_rate = float(np.mean(outside)) if outside else 0.0
    diff = inside_rate - outside_rate
    strength = strength_from_score(diff)
    total = finish_stats.get("total", 0)
    return {
        "name": "Course Bias",
        "state": band_from_score(diff, pos_thresh=0.05, neg_thresh=-0.05),
        "score": diff,
        "strength": strength,
        "detail": f"inside rate={inside_rate:.2f}, outside={outside_rate:.2f}, diff {diff:+.2f}",
        "coverage": f"n={total} historical races",
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
    strength = strength_from_score(diff)
    return {
        "name": "Field Imbalance",
        "state": band_from_score(diff, pos_thresh=0.2, neg_thresh=-0.2),
        "score": diff,
        "strength": strength,
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


def get_racer_id(row: pd.Series, lane: int) -> Optional[int]:
    return _safe_int(first_nonblank(
        row.get(f"racer_id_{lane}"),
        row.get(f"racer_id_{lane}_x"),
        row.get(f"racer_id_{lane}_y"),
        row.get(f"racerId_{lane}"),
        row.get(f"racerId_{lane}_x"),
        row.get(f"racerId_{lane}_y"),
    ))


def _weather_severity(row: pd.Series) -> float:
    wind = safe_float(first_nonblank(row.get("windPow"), row.get("wind_speed"))) or 0.0
    wave = safe_float(first_nonblank(row.get("waveHight"), row.get("wave_height"))) or 0.0
    return 0.1 * wind + 0.05 * wave


def _weather_bin(row: pd.Series) -> str:
    sev = _weather_severity(row)
    if sev <= 0.4:
        return "calm"
    if sev <= 0.9:
        return "moderate"
    return "rough"


def compute_racer_weather_stats(df: pd.DataFrame) -> Dict[int, Dict[str, Dict[str, Dict[str, float]]]]:
    """Aggregate per-racer top3/win rates by weather bin and wind sector."""
    if df.empty or "finish1_lane" not in df.columns:
        return {}

    stats: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for _, row in df.iterrows():
        bin_name = _weather_bin(row)
        sector = _wind_sector(row)
        f1, f2, f3 = row.get("finish1_lane"), row.get("finish2_lane"), row.get("finish3_lane")
        for lane in range(1, 7):
            rid = get_racer_id(row, lane)
            if rid is None:
                continue
            rstats = stats.setdefault(rid, {})
            bin_stats = rstats.setdefault(bin_name, {})

            def _update_bucket(key: str) -> None:
                bstats = bin_stats.setdefault(key, {"races": 0, "top3": 0, "win": 0})
                bstats["races"] += 1
                if f1 == lane:
                    bstats["win"] += 1
                    bstats["top3"] += 1
                elif f2 == lane or f3 == lane:
                    bstats["top3"] += 1

            # Update specific sector (or unknown) and an all-sector aggregate for fallback
            _update_bucket(sector or "unknown")
            _update_bucket("all")

    # Convert counts to rates
    for rid, bins in stats.items():
        for bin_name, sector_map in bins.items():
            for key, bstats in sector_map.items():
                races = bstats["races"]
                if races > 0:
                    bstats["top3_rate"] = bstats["top3"] / races
                    bstats["win_rate"] = bstats["win"] / races
                else:
                    bstats["top3_rate"] = None
                    bstats["win_rate"] = None

    return stats


def compute_weather_suitability_per_racer(row: pd.Series, racer_weather_stats: Dict[int, Dict[str, Dict[str, Dict[str, float]]]]) -> Dict[str, Any]:
    bin_name = _weather_bin(row)
    sector = _wind_sector(row)
    sector_key = sector or "unknown"

    scores: List[Optional[float]] = []
    top3_rates: List[Optional[float]] = []
    for lane in range(1, 7):
        rid = get_racer_id(row, lane)
        if rid is None:
            scores.append(None)
            top3_rates.append(None)
            continue
        bin_stats = racer_weather_stats.get(rid, {}).get(bin_name, {})
        bstats = bin_stats.get(sector_key) or bin_stats.get("all")
        if not bstats or bstats.get("top3_rate") is None:
            scores.append(None)
            top3_rates.append(None)
            continue
        top3_rate = bstats["top3_rate"]
        top3_rates.append(top3_rate)
    # Derive z-like score vs field for lanes with data
    for i in range(6):
        if top3_rates[i] is None:
            scores.append(None)
        else:
            scores.append(score_vs_field(top3_rates, i, invert=False))

    available = [s for s in scores if s is not None]
    if not available:
        return {
            "name": "Weather Suitability",
            "state": "Neutral",
            "detail": "no per-racer weather history",
            "coverage": "n=0",
        }

    avg_score = float(np.mean(available))
    strength = strength_from_score(avg_score)
    return {
        "name": "Weather Suitability",
        "state": band_from_score(avg_score, pos_thresh=0.1, neg_thresh=-0.1),
        "score": avg_score,
        "strength": strength,
        "detail": f"per-racer weather fit score {avg_score:.2f} (bin={bin_name}, sector={sector_key})",
        "coverage": f"lanes with data={len(available)}/6",
    }


def compute_factors_for_race(row: pd.Series, finish_stats: Dict[str, Any], racer_weather_stats: Dict[int, Dict[str, Dict[str, float]]]) -> List[Dict[str, Any]]:
    # Core factors
    start_factor = compute_start_stability(row)
    motor_factor = compute_motor_performance(row)
    racer_factor = compute_racer_consistency(row)
    weather_factor = compute_weather_suitability_per_racer(row, racer_weather_stats)
    course_factor = compute_course_bias(finish_stats)

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


def aggregate_race_signal(factors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Phase 3: Aggregate factors into race-level signal (Phase 3 Signal Structuring)."""
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    strong_positives = 0
    strong_negatives = 0
    
    for f in factors:
        state = f.get("state", "Neutral")
        counts[state] += 1
        strength = f.get("strength", "minimal")
        
        if state == "Positive" and strength in ("strong", "moderate"):
            strong_positives += 1
        elif state == "Negative" and strength in ("strong", "moderate"):
            strong_negatives += 1
    
    # Determine overall race sentiment
    if counts["Positive"] >= 4 and strong_positives >= 2:
        overall = "Highly Favorable"
    elif counts["Positive"] >= 3 and counts["Negative"] <= 1:
        overall = "Favorable"
    elif counts["Negative"] >= 4 or strong_negatives >= 3:
        overall = "Unfavorable"
    elif counts["Negative"] >= 2 and counts["Positive"] <= 1:
        overall = "Risky"
    else:
        overall = "Mixed Signals"
    
    return {
        "overall": overall,
        "positive_count": counts["Positive"],
        "neutral_count": counts["Neutral"],
        "negative_count": counts["Negative"],
        "strong_positives": strong_positives,
        "strong_negatives": strong_negatives,
    }


def _has_core_factor_data(row: pd.Series) -> bool:
    # Keep rows that have any usable start/motor/form metric to avoid Neutral n=0 spam
    def _any_values(prefix: str) -> bool:
        for i in range(1, 7):
            v = first_nonblank(row.get(f"{prefix}_{i}"), row.get(f"{prefix}_{i}_x"), row.get(f"{prefix}_{i}_y"))
            if v not in (None, ""):
                if not (isinstance(v, float) and np.isnan(v)):
                    return True
        return False

    return _any_values("ave_start_time") or _any_values("motor_place2Ratio") or _any_values("place2Ratio_national") or _any_values("win_rate_national")


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

    # Drop rows with zero usable factor inputs to avoid Neutral n=0 outputs for empty races
    df = df[df.apply(_has_core_factor_data, axis=1)]
    if df.empty:
        print("No races with factor data after filtering; aborting.")
        sys.exit(1)

    # Precompute finish-by-lane stats for this venue subset
    finish_stats = compute_finish_stats(df)

    # Precompute per-racer weather performance by weather bin for this venue subset
    racer_weather_stats = compute_racer_weather_stats(df)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["date", "raceNumber"], ascending=[False, False])

    recent = df.head(args.limit)

    for _, row in recent.iterrows():
        race_label = f"{row.get('date')} R{row.get('raceNumber')}"
        # Display the matched venue value or fall back to raw venue columns
        display_venue = row.get('venue_matched', row.get('venue', row.get('venue_x', 'unknown')))
        
        factors = compute_factors_for_race(row, finish_stats, racer_weather_stats)
        signal = aggregate_race_signal(factors)
        
        print(f"\n{'='*60}")
        print(f"Race: {race_label} (venue={display_venue})")
        print(f"{'='*60}")
        print(f"📊 RACE SIGNAL: {signal['overall']}")
        print(f"   (+{signal['positive_count']} Positive / ~{signal['neutral_count']} Neutral / -{signal['negative_count']} Negative)")
        print(f"   Strong factors: {signal['strong_positives']} positive, {signal['strong_negatives']} negative")
        print(f"\n🔍 TOP CONTRIBUTING FACTORS:")
        
        for i, f in enumerate(top_factors(factors, k=5), 1):
            state_icon = "✅" if f['state'] == "Positive" else "⚠️" if f['state'] == "Neutral" else "❌"
            strength = f.get('strength', 'minimal')
            print(f"   {i}. {state_icon} [{f['state']} ({strength})] {f['name']}")
            print(f"      {f['detail']}")
            print(f"      Coverage: {f['coverage']}")


if __name__ == "__main__":
    main()
