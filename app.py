# app.py
from __future__ import annotations

import traceback
import pandas as pd
import streamlit as st

# ここがあなたの既存コードに合わせて要調整（関数名だけ合わせる）
# 例: run_load_and_train.py に load_merged_race_df() がある前提
from run_load_and_train import load_merged_race_df, train_model  # <- 合わなければここだけ直す


st.set_page_config(page_title="Boatrace AI (MVP)", layout="wide")
st.title("Boatrace AI — 判断材料を増やす（MVP）")

# ========== Sidebar (設定) ==========
st.sidebar.header("設定")

target_venue = st.sidebar.text_input("対象競艇場（例：常滑）", value="常滑")

# 日付フィルタ（まずは “ロード後に絞る” のが安全）
use_date_filter = st.sidebar.checkbox("日付で絞る", value=False)
start_date = st.sidebar.date_input("開始日", value=pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("終了日", value=pd.to_datetime("2019-12-31"))

st.sidebar.divider()
run_load = st.sidebar.button("STEP 1: データを読み込む", type="primary")

# session_state でデータ保持
if "df" not in st.session_state:
    st.session_state.df = None

# ========== Helpers ==========
def normalize_venue(s: str) -> str:
    if s is None:
        return ""
    # 全角スペース・半角スペース除去
    return str(s).replace("　", "").replace(" ", "").strip()

def add_venue_clean(df: pd.DataFrame) -> pd.DataFrame:
    if "venue_clean" not in df.columns:
        # venue列名が違う場合はここを調整
        if "venue" in df.columns:
            df["venue_clean"] = df["venue"].map(normalize_venue)
        elif "place" in df.columns:
            df["venue_clean"] = df["place"].map(normalize_venue)
        else:
            raise KeyError("venue列が見つからない（venue / place がない）")
    return df

def apply_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    if not use_date_filter:
        return df
    # date列名が違う場合はここを調整
    if "date" not in df.columns:
        raise KeyError("date列が見つからない")
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    return d[(d["date"] >= sd) & (d["date"] <= ed)]

# ========== Load ==========
if run_load:
    try:
        with st.spinner("データ読み込み中..."):
            df = load_merged_race_df()  # <- あなたの既存関数に合わせる
            df = add_venue_clean(df)
            st.session_state.df = df
        st.success("読み込み完了")
    except Exception as e:
        st.session_state.df = None
        st.error("読み込みでエラー")
        st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))

df = st.session_state.df

# ========== Main view ==========
if df is None:
    st.info("左の「STEP 1: データを読み込む」を押してください。")
    st.stop()

# ここから “必ずチェックしてから学習” をUIで強制する
target_clean = normalize_venue(target_venue)

df_view = df
try:
    df_view = apply_date_filter(df_view)
except Exception as e:
    st.error("日付フィルタでエラー（date列の名前が違う可能性）")
    st.code(str(e))
    st.stop()

# 基本情報
col1, col2, col3 = st.columns(3)
col1.metric("総行数", f"{len(df_view):,}")
if "date" in df_view.columns:
    dmin = pd.to_datetime(df_view["date"], errors="coerce").min()
    dmax = pd.to_datetime(df_view["date"], errors="coerce").max()
    col2.metric("date range", f"{dmin} ~ {dmax}")
else:
    col2.metric("date range", "N/A（date列なし）")

tok_rows = 0
if "venue_clean" in df_view.columns:
    tok_rows = int((df_view["venue_clean"] == target_clean).sum())
col3.metric(f"{target_venue} 行数", f"{tok_rows:,}")

st.subheader("会場の件数（上位）")
top = df_view["venue_clean"].value_counts().head(15)
st.dataframe(top.rename_axis("venue_clean").reset_index(name="count"), use_container_width=True)

st.subheader("STEP 2: 学習（会場行数が0なら実行不可）")

# 0行ならボタンを無効化
disabled = (tok_rows == 0)

train_clicked = st.button("学習を開始する", disabled=disabled)

if disabled:
    st.warning(
        f"{target_venue} が 0 行です。学習できません。\n\n"
        "よくある原因:\n"
        "- 会場名が '常　滑' のようにスペース混在（このUIでは除去してます）\n"
        "- そもそも読み込んだDFに常滑が入っていない\n"
        "- date絞り込みで常滑が落ちている\n"
    )

if train_clicked:
    try:
        with st.spinner("学習中..."):
            # train_model 側で df を受け取れるなら渡すのがベスト
            # 受け取れないなら train_model() にして内部ロードでもOK
            result = train_model(df_view, target_clean)  # <- ここも既存に合わせる
        st.success("学習完了")
        st.write(result)
    except Exception as e:
        st.error("学習でエラー")
        st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
