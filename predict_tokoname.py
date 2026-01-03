import joblib
import numpy as np
import pandas as pd

# 学習済みモデルと特徴量リストを読み込み
model = joblib.load("tokoname_p1_model.pkl")
feature_cols = joblib.load("tokoname_p1_features.pkl")

def predict_p1_win_prob(features: dict) -> float:
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
    """
    # DataFrame にしてモデルに突っ込む
    df = pd.DataFrame([features], columns=feature_cols)
    prob = model.predict_proba(df)[0, 1]
    return float(prob)

if __name__ == "__main__":
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

    prob = predict_p1_win_prob(sample_features)
    print(f"このレースで1号艇が勝つ確率: {prob:.3f}（{prob*100:.1f}%）")
