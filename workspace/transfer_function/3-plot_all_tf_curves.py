import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from omu_eddy_covariance import TransferFunctionCalculator

# font_size_label: int = 16
# font_size_ticks: int = 14
# plt.rcParams["axes.labelsize"] = font_size_label
# plt.rcParams["xtick.labelsize"] = font_size_ticks
# plt.rcParams["ytick.labelsize"] = font_size_ticks
# 日本語フォントの設定
# plt.rcParams["font.family"] = "MS Gothic"

# ローカルフォントを読み込む場合
font_path: str = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
font_prop: FontProperties = FontProperties(fname=font_path)
font_array: list[str] = [font_prop.get_name(), "Dejavu Sans"]

# font_array: list[str] = ["Dejavu Sans"]

# rcParamsでの全体的な設定
plt.rcParams.update(
    {
        "font.family": font_array,
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
    }
)


def plot_all_tf_curves(
    file_path: str,
    output_dir: str | None = None,
    output_basename: str = "all_tf_curves",
    show_plot: bool = True,
):
    """
    指定されたCSVファイルからch4とc2h6の伝達関数を計算し、別々のグラフにプロットする関数。

    この関数は、与えられたCSVファイルから伝達関数の係数を読み込み、
    ch4とc2h6それぞれの伝達関数曲線を計算して別々のグラフにプロットします。
    各グラフには、全てのa値を用いた近似直線が描かれます。

    Args:
        file_path (str): 伝達関数の係数が格納されたCSVファイルのパス。

    Returns:
        None: この関数は結果をプロットするだけで、値を返しません。

    Note:
        - CSVファイルには 'Date', 'a_ch4-used' と 'a_c2h6-used' カラムが必要です。
        - プロットは対数スケールで表示され、グリッド線が追加されます。
        - 結果は plt.show() を使用して表示されます。
    """
    # CSVファイルを読み込む
    df = pd.read_csv(file_path)

    # ガスの種類とそれに対応する色のリスト
    gases = ["ch4", "c2h6"]
    # ガスの表示ラベル（LaTeX記法）
    gas_labels = ["CH$_4$", "C$_2$H$_6$"]
    base_colors = ["red", "orange"]

    # 各ガスについてプロット
    for gas, gas_label, base_color in zip(gases, gas_labels, base_colors):
        plt.figure(figsize=(10, 6))

        # 色のグラデーションを作成
        n_dates = len(df)
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_dates))

        # 全てのa値を用いて伝達関数をプロット
        for _, (row, color) in enumerate(zip(df.iterrows(), colors)):
            a = row[1][f"a_{gas}-used"]
            date = row[1]["Date"]
            x_fit = np.logspace(-3, 1, 1000)
            y_fit = TransferFunctionCalculator.transfer_function(x_fit, a)
            plt.plot(
                x_fit,
                y_fit,
                "--",  # 破線に変更
                color=color,
                alpha=0.7,
                label=f"{date} (a = {a:.3f})",
            )

        # 平均のa値を用いた伝達関数をプロット
        a_mean = df[f"a_{gas}-used"].mean()
        x_fit = np.logspace(-3, 1, 1000)
        y_fit = TransferFunctionCalculator.transfer_function(x_fit, a_mean)
        plt.plot(
            x_fit,
            y_fit,
            "-",
            color=base_color,
            linewidth=3,
            label=f"平均 (a = {a_mean:.3f})",
        )

        # グラフの設定
        plt.xscale("log")
        plt.xlabel("f (Hz)")
        plt.ylabel("コスペクトル比")
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.title(f"{gas_label}の伝達関数")

        # グラフの表示
        plt.tight_layout()

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path: str = os.path.join(output_dir, f"{output_basename}-{gas}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()


# メイン処理
try:
    tf_csv_path: str = "/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/TF_Ultra_a.csv"
    plot_all_tf_curves(
        file_path=tf_csv_path,
        output_dir="/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/outputs/tf",
        show_plot=False,
    )
except KeyboardInterrupt:
    # キーボード割り込みが発生した場合、処理を中止する
    print("KeyboardInterrupt occurred. Abort processing.")
