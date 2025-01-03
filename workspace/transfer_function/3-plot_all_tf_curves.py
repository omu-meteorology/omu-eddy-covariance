import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from omu_eddy_covariance import TransferFunctionCalculator


def plot_tf_curve(
    df: pd.DataFrame,
    date_key: str,
    coef_a_key: str,
    gas_label: str,
    base_color: str,
    output_dir: str | None = None,
    output_basename: str = "all_tf_curves",
    show_plot: bool = True,
    add_xlabel: bool = True,
    label_x: str = "f (Hz)",
    label_y: str = "無次元コスペクトル比",
    gas_name: str | None = None,  # 出力ファイル名用
):
    """
    伝達関数を計算し、グラフにプロットする関数。

    Args:
        df (pd.DataFrame): 伝達関数の係数が格納されたDataFrame
        date_key (str): 日付が格納されているカラムの名前
        coef_a_key (str): 係数が格納されているカラムの名前
        gas_label (str): プロットに表示するガスのラベル（例: "CH$_4$"）
        base_color (str): 平均値の線の色
        output_dir (str | None): 出力ディレクトリ。Noneの場合は保存しない
        output_basename (str): 出力ファイル名のベース
        show_plot (bool): プロットを表示するかどうか
        add_xlabel (bool): x軸ラベルを追加するかどうか
        label_x (str): x軸のラベル
        label_y (str): y軸のラベル
        gas_name (str | None): 出力ファイル名に使用するガス名。Noneの場合はcoef_a_keyを使用

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))

    # 色のグラデーションを作成
    n_dates = len(df)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_dates))

    # 全てのa値を用いて伝達関数をプロット
    for _, (row, color) in enumerate(zip(df.iterrows(), colors)):
        a = row[1][coef_a_key]
        date = row[1][date_key]
        x_fit = np.logspace(-3, 1, 1000)
        y_fit = TransferFunctionCalculator.transfer_function(x_fit, a)
        plt.plot(
            x_fit,
            y_fit,
            "--",
            color=color,
            alpha=0.7,
            label=f"{date} (a = {a:.3f})",
        )

    # 平均のa値を用いた伝達関数をプロット
    a_mean = df[coef_a_key].mean()
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
    label_y_formatted: str = f"{label_y}\n" f"({gas_label} / 顕熱)"
    plt.xscale("log")
    if add_xlabel:
        plt.xlabel(label_x)
    plt.ylabel(label_y_formatted)
    plt.legend(loc="lower left", fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        # 出力ファイル名用のガス名を決定
        output_gas_name = gas_name if gas_name is not None else coef_a_key
        output_path: str = os.path.join(
            output_dir, f"{output_basename}-{output_gas_name}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()


# メイン処理
if __name__ == "__main__":
    tf_csv_path: str = "/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/TF_Ultra_a.csv"
    output_dir: str = "/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/outputs/tf"
    # フォント名を指定
    font_array: list[str] = [
        # "MS Gothic",
        "IPAGothic",
        "Dejavu Sans",
    ]

    try:
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

        # CSVファイルを読み込む
        df = pd.read_csv(tf_csv_path)

        # ガスの種類とそれに対応する設定
        gas_configs = [
            ("a_ch4-used", "CH$_4$", "red", "ch4"),
            ("a_c2h6-used", "C$_2$H$_6$", "orange", "c2h6"),
        ]

        # 各ガスについてプロット
        for coef_a_key, gas_label, base_color, gas_name in gas_configs:
            plot_tf_curve(
                df=df,
                date_key="Date",
                coef_a_key=coef_a_key,
                gas_label=gas_label,
                base_color=base_color,
                gas_name=gas_name,
                output_dir=output_dir,
                show_plot=False,
            )

    except KeyboardInterrupt:
        print("KeyboardInterrupt occurred. Abort processing.")
