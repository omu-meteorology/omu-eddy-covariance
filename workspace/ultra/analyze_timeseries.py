import pandas as pd
from datetime import timedelta
from omu_eddy_covariance import EddyDataPreprocessor


def classify_source(ratio):
    """メタンの発生源を比から分類する関数"""
    if ratio >= 100:
        return "comb"
    elif ratio >= 5:
        return "gas"
    else:
        return "bio"


def analyze_methane_ratio(df, window_seconds=300):
    """
    メタンの発生源を解析する関数

    Args:
        df (pd.DataFrame): 解析対象のデータフレーム
        window_seconds (int): 移動平均の窓サイズ（秒）。デフォルトは300秒。

    Returns:
        tuple[pd.DataFrame, pd.Series]: 解析結果のデータフレームと発生源の集計結果
    """
    # データフレームのタイムスタンプを datetime 型に変換
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df.set_index("TIMESTAMP", inplace=True)

    # 移動平均の計算（300秒間）
    window_size = int(window_seconds * 10)  # 0.1秒間隔のデータなので10倍

    # CH4とC2H6の移動平均を計算
    df["ch4_mv"] = (
        df["Ultra_CH4_ppm_C"]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
    )
    df["c2h6_mv"] = (
        df["Ultra_C2H6_ppb"]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
    )

    # 移動平均からの偏差を計算
    df["ch4_delta"] = df["Ultra_CH4_ppm_C"] - df["ch4_mv"]
    df["c2h6_delta"] = df["Ultra_C2H6_ppb"] - df["c2h6_mv"]

    # CH4とC2H6の移動相関を計算
    df["ch4_c2h6_correlation"] = (
        df["Ultra_CH4_ppm_C"]
        .rolling(window=window_size, min_periods=1)
        .corr(df["Ultra_C2H6_ppb"])
    )

    # 10秒間隔でデータをグループ化して解析
    sampling_seconds = 10
    # sampling_seconds = 5
    df["time_window"] = df.index.floor(f"{sampling_seconds}S")

    results = []
    for window_start, group in df.groupby("time_window"):
        # CH4とC2H6の変動量を計算
        ch4_delta = group["ch4_delta"].mean()
        c2h6_delta = group["c2h6_delta"].mean()
        correlation = group["ch4_c2h6_correlation"].mean()

        # 変動比の計算（ゼロ除算を防ぐ）
        if abs(c2h6_delta) > 1e-10:  # 十分小さい値で判定
            ratio = abs(ch4_delta / c2h6_delta)
        else:
            ratio = 0

        # 結果を保存
        results.append(
            {
                "window_start": window_start,
                "window_end": window_start + timedelta(seconds=sampling_seconds),
                "ch4_delta": ch4_delta,
                "c2h6_delta": c2h6_delta,
                "correlation": correlation,
                "ratio": ratio,
                "source": classify_source(ratio),
            }
        )

    # 結果をデータフレームに変換
    results_df = pd.DataFrame(results)

    # 発生源ごとの集計
    source_summary = results_df["source"].value_counts()

    return results_df, source_summary


def main():
    # データ読み込み
    filepath = "/home/connect0459/labo/omu-eddy-covariance/workspace/ultra/private/data/2024.10.10/eddy_csv/TOA5_37477.SAC_Ultra.Eddy_107_2024_10_10_1000.dat"
    edp = EddyDataPreprocessor()
    df, _ = edp.get_resampled_df(filepath=filepath)

    # 分析実行
    results, summary = analyze_methane_ratio(df)

    # 結果表示
    print("\n=== 時間窓ごとの分析結果 ===")
    print(results)
    print("\n=== 発生源の集計結果 ===")
    print(summary)

    # 結果をCSVファイルとして保存
    results.to_csv(
        "/home/connect0459/labo/omu-eddy-covariance/workspace/ultra/private/data/2024.10.10/eddy_csv/TOA5_37477.SAC_Ultra.Eddy_107_2024_10_10_1400.dat",
        index=False,
    )


if __name__ == "__main__":
    main()
