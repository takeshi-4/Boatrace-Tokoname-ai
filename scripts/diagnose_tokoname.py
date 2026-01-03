#!/usr/bin/env python3
import os
import sys
import glob
import unicodedata
import argparse
import pandas as pd

# Add boatrace-master to path (works when run from repo root OR from scripts/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOATRACE_PATH = os.path.join(REPO_ROOT, "boatrace-master")
if BOATRACE_PATH not in sys.path:
    sys.path.insert(0, BOATRACE_PATH)

from src.data_preparing.loader import make_race_result_df

def normalize_venue(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = unicodedata.normalize("NFKC", str(x))
    return "".join(ch for ch in s if not ch.isspace()).strip()

def col_missing_report(df, cols):
    total = len(df)
    rows = []
    for c in cols:
        if c in df.columns:
            nulls = df[c].isna().sum()
            rows.append((c, int(nulls), total, float(nulls)/total if total else 0.0))
        else:
            rows.append((c, total, total, 1.0))
    return rows

def print_nonnull_examples(df, col, show=5):
    if col not in df.columns:
        print(f"  {col}: <missing column>")
        return
    sub = df[df[col].notnull()][["date", "raceNumber", col]]
    cnt = len(sub)
    print(f"  {col}: non-null count = {cnt}")
    if cnt:
        print(sub.head(show).to_string(index=False))


def debug_raw_txt_weather():
    """Debug: read a raw TXT file and show what the parser would extract."""
    import glob as g
    files = sorted(g.glob(os.path.join(REPO_ROOT, "boatrace-master", "data", "results_race", "K*.TXT")))
    if not files:
        return
    
    filename = files[0]  # Take first file
    print(f"\n=== DEBUG: Raw TXT parsing test from {os.path.basename(filename)} ===")
    
    with open(filename, "r", encoding="shift_jis", errors="replace") as f:
        text = f.read()
    
    import re as regex
    blocks = regex.split(r"[0-9][0-9]KBGN\n", text)
    blocks = blocks[1:]
    
    count = 0
    for block in blocks[:1]:  # Just first block
        parts = block.split("\n\n\n")
        parts = [p.splitlines() for p in parts if p.strip()]
        
        jcd_body = parts[1:-1]
        for race_body in jcd_body[:2]:  # First 2 races
            if len(race_body) > 1:
                print(f"  race_body[1] = {repr(race_body[1][:100])}")
                # Test parsing
                from src.data_preparing.loader import _parse_weather_line
                result = _parse_weather_line(race_body[1])
                print(f"  parsed: {result}")
            count += 1
            if count >= 2:
                break

def inspect_beforeinfo_files(beforeinfo_glob, target_clean, cols):
    print("\nScanning beforeinfo files:")
    for fn in sorted(glob.glob(beforeinfo_glob)):
        try:
            df = pd.read_csv(fn)
        except Exception as e:
            print(f"  SKIP {fn}: read error {e}")
            continue
        if "venue_clean" not in df.columns and "venue" in df.columns:
            df["venue_clean"] = df["venue"].map(normalize_venue)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df_target = df[df["venue_clean"] == target_clean] if "venue_clean" in df.columns else df[df.get("venue", "").map(normalize_venue)==target_clean]
        if df_target.empty:
            continue
        print(f"  {os.path.basename(fn)}: rows for {target_clean} = {len(df_target)}")
        for c in cols:
            if c in df_target.columns:
                print(f"    {c}: non-null={int(df_target[c].notnull().sum())}")
            else:
                print(f"    {c}: <no column>")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", default="常滑")
    ap.add_argument("--beforeinfo-glob", default=None)
    args = ap.parse_args()

    target = normalize_venue(args.venue)

    # Debug raw TXT parsing
    debug_raw_txt_weather()

    # Load merged df
    df = make_race_result_df()

    if df is None or df.empty:
        print("Merged dataframe is empty.")
        return

    # Ensure venue_clean exists
    if "venue_clean" not in df.columns and "venue" in df.columns:
        df["venue_clean"] = df["venue"].map(normalize_venue)

    print("date range:", df["date"].min(), df["date"].max())
    print("rows:", len(df))
    print("dup check key(date,venue_clean,raceNumber):", df.duplicated(subset=["date","venue_clean","raceNumber"]).sum())

    tok = df[df["venue_clean"] == target]
    print(f"\nRows for '{target}': {len(tok)}")
    if tok.empty:
        print("\nDistinct venue -> venue_clean mapping (sample):")
        if "venue" in df.columns:
            m = df[["venue"]].drop_duplicates().head(200)
            m["venue_clean"] = m["venue"].map(normalize_venue)
            print(m.to_string(index=False))
        return

    weather_cols = ["weather","windDir","windPow","waveHight","ruler","wind_speed","wave_height","temperature"]
    print("\nMissingness report for weather/condition columns (target subset):")
    for c, nulls, total, frac in col_missing_report(tok, weather_cols):
        print(f"  {c}: nulls={nulls}/{total} ({frac:.2%})")

    print("\nShow up to 5 non-null examples per column:")
    for c in weather_cols:
        print_nonnull_examples(tok, c, show=5)
        print("")

    # Scan beforeinfo files if available to find which files have non-null data
    default_beforeinfo = os.path.join(REPO_ROOT, "boatrace-master", "data", "beforeinfo", "*.csv")
    beforeinfo_glob = args.beforeinfo_glob or default_beforeinfo
    inspect_beforeinfo_files(beforeinfo_glob, target, weather_cols)

if __name__ == "__main__":
    main()