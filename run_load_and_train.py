import os
import sys
import argparse
import unicodedata
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "boatrace-master"))

from src.data_preparing.loader import make_race_result_df


# -----------------------------
# 共通ユーティリティ
# -----------------------------
def normalize_venue(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = unicodedata.normalize("NFKC", str(x))
    return "".join(ch for ch in s if not ch.isspace()).strip()


def _row_value(row, col):
    if col not in row.index:
        return None
    v = row[col]
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return None if s == "" or s.lower() == "nan" else v


def _first_value(row, candidates):
    for c in candidates:
        v = _row_value(row, c)
        if v is not None:
            return v, c
    return None, None


# -----------------------------
# データロード
# -----------------------------
def load_merged_race_df():
    txt_glob = os.path.join(
        BASE_DIR, "boatrace-master", "data", "results_race", "K1*.TXT"
    )
    print("[TXT GLOB]", txt_glob)

    df = make_race_result_df(race_results_file_path=txt_glob)

    if df is None or df.empty:
        return pd.DataFrame()

    # PerformanceWarning 対策（デフラグ）
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    if "venue_clean" not in df.columns and "venue" in df.columns:
        df["venue_clean"] = df["venue"].map(normalize_venue)

    return df


# -----------------------------
# 1レース説明（判断材料AIの最小単位）
# -----------------------------
def explain_one_race(df, venue_clean, date_str, race_no, debug_one=False):
    d = pd.to_datetime(date_str)

    sub = df[
        (df["venue_clean"] == venue_clean)
        & (df["date"].dt.date == d.date())
        & (df["raceNumber"] == race_no)
    ]

    if sub.empty:
        raise ValueError(
            f"no row found for venue={venue_clean}, date={date_str}, race={race_no}"
        )

    row = sub.iloc[0]

    # --- デバッグ：列の実在確認 ---
    if debug_one:
        cols = [c for c in sub.columns if "racer" in c]
        print("[DEBUG one-row columns.T]")
        print(sub.iloc[[0]][cols].T)

    # --- 判断材料 ---
    factors = []

    def add_factor(name, col):
        v = _row_value(row, col)
        if v is not None:
            factors.append({"factor": name, "value": v, "source": col})

    add_factor("天候", "weather")
    add_factor("風向", "windDir")
    add_factor("風速", "windPow")
    add_factor("波高", "waveHight")
    add_factor("決まり手", "ruler")
    add_factor("気温", "temperature")
    add_factor("直前風速", "wind_speed")
    add_factor("直前波高", "wave_height")

    # --- 出走者 ---
    racers = []
    for frame in range(1, 7):
        # racerId（確定版）
        rid, rid_src = _first_value(
            row,
            [
                f"racerId_{frame}",
                f"racer_id_{frame}",
                f"racerID_{frame}",
            ],
        )

        # racerName（確定版）
        rnm, rnm_src = _first_value(
            row,
            [
                f"racerName_{frame}",
                f"racer_{frame}_x",
                f"racer_{frame}_y",
            ],
        )

        if rid is None and rnm is None:
            continue

        st, _ = _first_value(row, [f"startTime_{frame}", f"start_time_{frame}"])
        ex, _ = _first_value(row, [f"exhibitionTime_{frame}", f"exhibition_time_{frame}"])
        mot, _ = _first_value(row, [f"motor_{frame}", f"motorNo_{frame}"])
        boat, _ = _first_value(row, [f"boat_{frame}", f"boatNo_{frame}"])
        crs, _ = _first_value(row, [f"cource_{frame}", f"course_{frame}"])

        try:
            rid_norm = int(str(rid)) if rid is not None else None
        except Exception:
            rid_norm = rid

        racers.append(
            {
                "frame": frame,
                "racerId": rid_norm,
                "racerName": None if rnm is None else str(rnm).strip(),
                "startTime": st,
                "exhibitionTime": ex,
                "course": crs,
                "motor": mot,
                "boat": boat,
                "sources": {"racerId": rid_src, "racerName": rnm_src},
            }
        )

    return {
        "venue": venue_clean,
        "date": str(pd.to_datetime(row["date"])),
        "raceNumber": int(row["raceNumber"]),
        "top_factors": factors[:5],
        "all_factors": factors,
        "racers": racers,
    }


# -----------------------------
# 集計（判断材料の分布）
# -----------------------------
def train_model(df, target_clean):
    tok = df[df["venue_clean"] == target_clean]
    if tok.empty:
        raise ValueError(f"Target venue '{target_clean}' contains 0 rows.")

    factors = {}
    for c in [
        "windDir",
        "windPow",
        "waveHight",
        "ruler",
        "wind_speed",
        "wave_height",
        "temperature",
    ]:
        if c in tok.columns:
            factors[c] = tok[c].value_counts(dropna=False).head(5).to_dict()

    return {
        "target_venue": target_clean,
        "rows": int(len(tok)),
        "date_range": [str(tok["date"].min()), str(tok["date"].max())],
        "top_factors": factors,
    }


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", default="常滑")
    ap.add_argument("--start-date")
    ap.add_argument("--end-date")
    ap.add_argument("--one-date")
    ap.add_argument("--one-race", type=int)
    ap.add_argument("--debug-one", action="store_true")
    args = ap.parse_args()

    df = load_merged_race_df()
    if df.empty:
        print("[STOP] merged dataframe is empty.")
        return

    print("date range:", df["date"].min(), df["date"].max())
    print("rows:", len(df))

    # 重複キー確認（merge事故検知）
    dup = df.duplicated(subset=["date", "venue_clean", "raceNumber"]).sum()
    print("dup check key(date,venue_clean,raceNumber):", dup)

    target_clean = normalize_venue(args.venue)

    tok = df[df["venue_clean"] == target_clean]
    print("tokoname rows (before date filter):", len(tok))
    if tok.empty:
        print(df["venue_clean"].value_counts().head(20))
        return

    if args.start_date or args.end_date:
        start = pd.to_datetime(args.start_date) if args.start_date else df["date"].min()
        end = pd.to_datetime(args.end_date) if args.end_date else df["date"].max()
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        print("rows after date filter:", len(df))

    tok = df[df["venue_clean"] == target_clean]
    print("tokoname rows (after date filter):", len(tok))

    if args.one_date and args.one_race:
        out = explain_one_race(
            df,
            target_clean,
            args.one_date,
            args.one_race,
            debug_one=args.debug_one,
        )
        print(out)
        return

    out = train_model(df, target_clean)
    print(out)


if __name__ == "__main__":
    main()
