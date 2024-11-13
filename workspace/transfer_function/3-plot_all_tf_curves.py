import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from omu_eddy_covariance import TransferFunctionCalculator

font_size_label: int = 16
font_size_ticks: int = 14
plt.rcParams["axes.labelsize"] = font_size_label
plt.rcParams["xtick.labelsize"] = font_size_ticks
plt.rcParams["ytick.labelsize"] = font_size_ticks
# 日本語フォントの設定
plt.rcParams["font.family"] = "MS Gothic"


def plot_all_tf_curves(file_path: str):
    """
    指定されたCSVファイルからCH4とC2H6の伝達関数を計算し、別々のグラフにプロットする関数。

    この関数は、与えられたCSVファイルから伝達関数の係数を読み込み、
    CH4とC2H6それぞれの伝達関数曲線を計算して別々のグラフにプロットします。
    各グラフには、全てのa値を用いた近似直線が描かれます。

    Args:
        file_path (str): 伝達関数の係数が格納されたCSVファイルのパス。

    Returns:
        None: この関数は結果をプロットするだけで、値を返しません。

    Note:
        - CSVファイルには 'Date', 'a_CH4-used' と 'a_C2H6-used' カラムが必要です。
        - プロットは対数スケールで表示され、グリッド線が追加されます。
        - 結果は plt.show() を使用して表示されます。
    """
    # CSVファイルを読み込む
    df = pd.read_csv(file_path)

    # ガスの種類とそれに対応する色のリスト
    gases = ["CH4", "C2H6"]
    colors = ["blue", "red"]

    # 各ガスについてプロット
    for gas, color in zip(gases, colors):
        plt.figure(figsize=(10, 6))

        # 全てのa値を用いて伝達関数をプロット
        for index, row in df.iterrows():
            a = row[f"a_{gas}-used"]
            date = row["Date"]
            x_fit = np.logspace(-3, 1, 1000)
            y_fit = TransferFunctionCalculator.transfer_function(x_fit, a)
            plt.plot(
                x_fit, y_fit, "-", color=color, alpha=0.3, label=f"{date} (a = {a:.4f})"
            )

        # 平均のa値を用いた伝達関数をプロット
        a_mean = df[f"a_{gas}-used"].mean()
        x_fit = np.logspace(-3, 1, 1000)
        y_fit = TransferFunctionCalculator.transfer_function(x_fit, a_mean)
        plt.plot(
            x_fit,
            y_fit,
            "-",
            color="black",
            linewidth=2,
            label=f"平均 (a = {a_mean:.4f})",
        )

        # グラフの設定
        plt.xscale("log")
        plt.xlabel("f (Hz)")
        plt.ylabel("コスペクトル比")
        plt.legend(loc="lower left", fontsize=8)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.title(f"{gas}の伝達関数")

        # グラフの表示
        plt.tight_layout()
        plt.show()


# メイン処理
try:
    tf_csv_path: str = (
        "C:\\Users\\nakao\\workspace\\sac\\ultra\\transfer_function\\tf-a\\TF_Ultra_a.csv"
    )
    plot_all_tf_curves(tf_csv_path)
except KeyboardInterrupt:
    # キーボード割り込みが発生した場合、処理を中止する
    print("KeyboardInterrupt occurred. Abort processing.")
