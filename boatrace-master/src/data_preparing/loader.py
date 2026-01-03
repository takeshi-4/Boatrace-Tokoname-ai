import glob
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import logging
import json

# Module-level diagnostics that callers/tests can inspect after a run
LAST_RUN_DIAGNOSTICS: dict = {}

# Logger setup (library-safe)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _normalize_venue_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("　", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
    )


def _safe_int(s: Any) -> Optional[int]:
    """Convert to int safely. Returns None for empty/NaN/invalid."""
    if s is None:
        return None
    try:
        ss = str(s).strip()
        if ss == "" or ss.lower() == "nan":
            return None
        return int(float(ss))  # handles "4731.0" as well
    except Exception:
        return None


def _is_blank(x: Any) -> bool:
    if x is None:
        return True
    s = str(x)
    return s.strip() == "" or s.strip().lower() == "nan"


def _pick_first_nonblank(*vals: Any) -> Optional[str]:
    for v in vals:
        if not _is_blank(v):
            return str(v).strip()
    return None


def _parse_weather_line(line: str) -> Dict[str, Optional[str]]:
    """
    Parse weather info from race header line like:
    '1R       ○@●@                 H1800m  曇り  風  北西  4m  波@  3cm'
    
    Returns dict with keys: weather, windDir, windPow, waveHight (all Optional[str])
    """
    result = {"weather": None, "windDir": None, "windPow": None, "waveHight": None}
    
    if not line or len(line) < 10:
        return result
    
    try:
        tokens = line.split()
        
        for i, tok in enumerate(tokens):
            # Weather: look for keywords like 晴れ, 曇り, 雨, 霧, etc.
            if any(w in tok for w in ['晴', '曇', '雨', '霧']):
                result["weather"] = tok.strip()
            
            # Wind direction: usually follows '風' keyword
            if tok == '風' and i + 1 < len(tokens):
                result["windDir"] = tokens[i + 1].strip()
            
            # Wind power: find number followed by 'm' (but NOT 'cm')
            if tok.endswith('m') and not tok.endswith('cm'):
                num_part = tok[:-1]  # Remove 'm'
                if num_part and num_part[0].isdigit():
                    result["windPow"] = num_part.strip()
            
            # Wave height: token ending with 'cm'
            if tok.endswith('cm') and any(c.isdigit() for c in tok):
                num_part = tok[:-2]  # Remove 'cm'
                if num_part:
                    result["waveHight"] = num_part.strip()
    except Exception:
        pass
    
    return result


def get_racers_from_row(row: pd.Series) -> List[Dict[str, Any]]:
    """
    (2) racers:[] 対策：列名ズレ/NaN/merge由来の _x/_y を吸収して必ず拾う。
    想定して拾う順：
      racerName_{i} / racerName_{i}_x / racerName_{i}_y / racer_{i} / racer_{i}_x / racer_{i}_y
      racerId_{i} / racerId_{i}_x / racerId_{i}_y / racer_id_{i} / racer_id_{i}_x / racer_id_{i}_y
    """
    racers: List[Dict[str, Any]] = []
    for i in range(1, 7):
        name = _pick_first_nonblank(
            row.get(f"racerName_{i}"),
            row.get(f"racerName_{i}_x"),
            row.get(f"racerName_{i}_y"),
            row.get(f"racer_{i}"),
            row.get(f"racer_{i}_x"),
            row.get(f"racer_{i}_y"),
        )
        rid_raw = _pick_first_nonblank(
            row.get(f"racerId_{i}"),
            row.get(f"racerId_{i}_x"),
            row.get(f"racerId_{i}_y"),
            row.get(f"racer_id_{i}"),
            row.get(f"racer_id_{i}_x"),
            row.get(f"racer_id_{i}_y"),
        )
        rid = _safe_int(rid_raw)

        if name is None:
            continue

        racers.append(
            {
                "frame": i,
                "racerId": rid,
                "racerName": name,
                "sources": {
                    "racerId": f"(best-effort) racerId_{i}/racer_id_{i}/_x/_y",
                    "racerName": f"(best-effort) racerName_{i}/racer_{i}/_x/_y",
                },
            }
        )
    return racers


def load_race_results(
    race_results_file_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../data/results_race/K1*.TXT",
    )
):
    race_result_dict = {
        "date": [],
        "venue": [],
        "raceNumber": [],
        "weather": [],
        "windDir": [],
        "windPow": [],
        "waveHight": [],
        "ruler": [],
    }

    for i in range(1, 7):
        # 生データ由来の列（元の列名を維持）
        race_result_dict[f"racerName_{i}"] = []
        race_result_dict[f"racerId_{i}"] = []

        # (2) 下流で揺れない “正規化済み” 列も追加（列名ズレ吸収用）
        race_result_dict[f"racer_{i}"] = []
        race_result_dict[f"racer_id_{i}"] = []

    files = sorted(glob.glob(race_results_file_path))
    for filename in files:
        logger.info("[LOADING TXT] %s", filename)

        m = re.search(r"K(\d{2})(\d{2})(\d{2})", os.path.basename(filename))
        if not m:
            continue
        file_date = datetime.strptime(
            f"20{m.group(1)}-{m.group(2)}-{m.group(3)}", "%Y-%m-%d"
        )

        with open(filename, "r", encoding="shift_jis", errors="replace") as f:
            text = f.read()

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        blocks = re.split(r"[0-9][0-9]KBGN\n", text)
        blocks = blocks[1:]

        skipped_blocks = []
        for block_index, block in enumerate(blocks):
            try:
                parts = block.split("\n\n\n")
                parts = [p.splitlines() for p in parts if p.strip()]
                if len(parts) < 2:
                    continue

                jcd_head = parts[0]
                jcd_body = parts[1:-1]
                if not jcd_head:
                    continue

                venue_raw = jcd_head[0][0:3]
                venue = venue_raw

                for race_body in jcd_body:
                    # (3) SKIP RACE の洪水対策 + レース番号の堅牢化
                    if not race_body or len(race_body) < 10:
                        continue

                    race_head = race_body[0] if race_body else ""
                    race_str = (race_head[2:4] if len(race_head) >= 4 else "").strip()
                    race = _safe_int(race_str)
                    if race is None:
                        continue

                    # Parse weather: find the line containing 'H1800m' or 'H1200m' (the race header with conditions)
                    # It could be in race_body[0] or race_body[1] depending on file format
                    weather_info = {"weather": None, "windDir": None, "windPow": None, "waveHight": None}
                    for line in race_body[:3]:  # Check first 3 lines
                        if 'H' in line and ('m' in line or 'cm' in line):
                            weather_info = _parse_weather_line(line)
                            if weather_info["weather"] or weather_info["windDir"] or weather_info["windPow"]:
                                break  # Found weather data, stop looking

                    race_result_dict["date"].append(file_date)
                    race_result_dict["venue"].append(venue)
                    race_result_dict["raceNumber"].append(race)

                    race_result_dict["weather"].append(weather_info["weather"])
                    race_result_dict["windDir"].append(weather_info["windDir"])
                    race_result_dict["windPow"].append(weather_info["windPow"])
                    race_result_dict["waveHight"].append(weather_info["waveHight"])
                    race_result_dict["ruler"].append(None)

                    racers_result = race_body[3:9]
                    tmp: Dict[int, Tuple[str, str]] = {i: ("", "") for i in range(1, 7)}

                    for r in racers_result:
                        if not r:
                            continue
                        frame = _safe_int(r[6] if len(r) > 6 else "")
                        if frame is None or frame not in tmp:
                            continue

                        rid = (r[8:12] if len(r) >= 12 else "").strip()
                        name = (r[13:21] if len(r) >= 21 else "").strip()
                        tmp[frame] = (name, rid)

                    for i in range(1, 7):
                        name_i, rid_i = tmp[i]
                        race_result_dict[f"racerName_{i}"].append(name_i)
                        race_result_dict[f"racerId_{i}"].append(rid_i)

                        # (2) 正規化列
                        race_result_dict[f"racer_{i}"].append(name_i)
                        race_result_dict[f"racer_id_{i}"].append(_safe_int(rid_i))
            except Exception as e:
                # Record the broken block and continue processing rest
                logger.exception("Error parsing block %s in file %s", block_index, filename)
                skipped_blocks.append({
                    "filename": filename,
                    "file_date": file_date.isoformat() if isinstance(file_date, datetime) else str(file_date),
                    "block_index": block_index,
                    "venue": venue if 'venue' in locals() else None,
                    "error": str(e),
                })
                continue

    df = pd.DataFrame(race_result_dict)

    # Save diagnostics for this run so callers can inspect
    LAST_RUN_DIAGNOSTICS['skipped_blocks'] = skipped_blocks
    LAST_RUN_DIAGNOSTICS['loaded_files'] = files

    # (1) PerformanceWarning / fragmentation 対策（最小修正）
    # df["venue_clean"] の追加より先に copy() してデフラグする
    df = df.copy()
    df["venue_clean"] = _normalize_venue_series(df["venue"])

    logger.info("Loaded %d rows from TXT files (%d skipped blocks)", len(df), len(skipped_blocks))

    return df


def load_race_results_supplementary_data(path):
    dfs = []
    for filename in glob.glob(path):
        dfs.append(pd.read_csv(filename))
    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # supplementary 側にも venue_clean を作る（merge key に必要）
    if "venue" in df.columns:
        df = df.copy()
        df["venue_clean"] = _normalize_venue_series(df["venue"])

    return df


def make_race_result_df(
    race_results_file_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../data/results_race/K1*.TXT",
    ),
    racelist_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../data/racelist/1*.csv",
    ),
    beforeinfo_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../data/beforeinfo/1*.csv",
    ),
):
    df = load_race_results(race_results_file_path)
    print(f"[DEBUG] After load_race_results: 'weather' in df = {'weather' in df.columns}")
    print(f"[DEBUG] Columns after load: {df.columns.tolist()}")

    racelist_df = load_race_results_supplementary_data(racelist_path)
    beforeinfo_df = load_race_results_supplementary_data(beforeinfo_path)

    key = ["date", "venue_clean", "raceNumber"]

    # Preserve weather columns from TXT (from df) by storing them before merge
    weather_cols = ["weather", "windDir", "windPow", "waveHight"]
    weather_data = df[weather_cols].copy() if all(c in df.columns for c in weather_cols) else None
    print(f"[DEBUG] weather_data is not None: {weather_data is not None}")

    if not racelist_df.empty:
        df = pd.merge(df, racelist_df, how="left", on=key)

    if not beforeinfo_df.empty:
        df = pd.merge(df, beforeinfo_df, how="left", on=key)

    # Restore weather columns from TXT (drop any CSV duplicates suffixed with _x or _y)
    print(f"[DEBUG] Before restoration: 'weather' in df = {'weather' in df.columns}")
    print(f"[DEBUG] Columns before restoration: {df.columns.tolist()}")
    if weather_data is not None:
        for col in weather_cols:
            # Always restore TXT version (create or overwrite the column)
            try:
                df[col] = weather_data[col].values
            except Exception:
                # fallback: align by index/labels if positional assign fails
                df[col] = weather_data[col]
            # Remove any suffixed versions from merge (e.g., 'weather_x', 'weather_y')
            for suffix in ['_x', '_y']:
                col_suff = col + suffix
                if col_suff in df.columns:
                    df = df.drop(columns=[col_suff])
    print(f"[DEBUG] After restoration: 'weather' in df = {'weather' in df.columns}")
    print(f"[DEBUG] Columns after restoration: {df.columns.tolist()}")

    # merge 後も念のためデフラグ（この後に特徴量を大量追加するなら効果あり）
    df = df.copy()

    return df
