import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def plot_stacked_flux(
    input_filepath: str,
    output_dir: str,
    concentration_unit: str = "nano",
    figsize: tuple[float, float] = (20, 13),
    output_basename: str = "ch4_flux_stacked_bar_directions",
    tag: str = "default",
    ylim: float | None = None,
):
    flux_unit: str = "nmol m$^{-2}$ s$^{-1}$"
    flux_magnification: float = 1
    if concentration_unit == "micro":
        flux_unit = "μmol m$^{-2}$ s$^{-1}$"
        flux_magnification = 1 / 1000
    elif concentration_unit != "nano":
        raise ValueError(
            "concentration_unitには`micro`または`nano`を指定する必要があります。"
        )

    # データの読み込み
    df: pd.DataFrame = pd.read_csv(input_filepath, skiprows=[1])

    # 方角の配置順序を定義（左上から時計回り）
    directions_order: list[str] = ["nw", "ne", "sw", "se"]
    titles: dict[str, str] = {"nw": "北西", "ne": "北東", "sw": "南西", "se": "南東"}

    # サブプロットを含む大きな図を作成
    fig = plt.figure(figsize=figsize)

    # 各方角についてサブプロットを作成
    for idx, direction in enumerate(directions_order, 1):
        # サブプロットの位置を設定
        ax = fig.add_subplot(2, 2, idx)

        # 文字列を数値に変換
        diurnal = pd.to_numeric(df[f"diurnal_{direction}"], errors="coerce")
        gasratio = pd.to_numeric(df[f"gasratio_{direction}"], errors="coerce")

        # 単位によって倍率を補正
        diurnal *= flux_magnification

        # diurnalが10以下のデータをマスク
        valid_mask = diurnal > 10

        # gas由来とbio由来のCH4フラックスを計算（信頼性の低いデータは0に設定）
        gas = np.where(valid_mask, diurnal * gasratio / 100, 0)
        bio = np.where(valid_mask, diurnal * (100 - gasratio) / 100, 0)

        # 積み上げ棒グラフの作成
        width = 0.8
        ax.bar(df["month"], gas, width, label="都市", color="orange")
        ax.bar(df["month"], bio, width, bottom=gas, label="生物", color="lightblue")

        # x軸の設定
        ax.set_xticks(df["month"])  # すべての月を目盛りとして設定
        ax.set_xticklabels(df["month"])  # すべての月をラベルとして表示

        # y軸の上限を設定
        if ylim is not None:
            ax.set_ylim(0, ylim)

        # gas比率の表示（信頼性の低いデータは表示しない）
        for i, (g, b, is_valid) in enumerate(zip(gas, bio, valid_mask)):
            if is_valid:
                total = g + b
                ratio = g / total * 100
                ax.text(
                    df["month"][i], total, f"{ratio:.0f}%", ha="center", va="bottom"
                )

        # グラフの装飾
        ax.set_title(titles[direction])

        # 凡例は1回だけ表示（右上のグラフに配置）
        if idx == 2:  # 右上のグラフ
            ax.legend(bbox_to_anchor=(0.95, 1), loc="upper right")

    # サブプロット間の間隔を調整（軸ラベル用のスペースを確保）
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    # 共通の軸ラベルを追加（figureの余白部分に配置）
    fig.text(
        0.5,
        0.02,
        "Month",
        ha="center",
        va="center",
    )
    fig.text(
        0.02,
        0.5,
        f"CH$_4$ Flux ({flux_unit})",
        va="center",
        rotation="vertical",
    )

    # グラフの保存
    plt.savefig(
        f"{output_dir}/{output_basename}-{tag}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


"""
Ubuntu環境でのフォントの手動設定
不要な方はコメントアウトして実行してください。
ここでは日本語フォントを読み込んでいます。

1. インストール : `sudo apt update && sudo apt install -y fonts-ipafont`
2. キャッシュ削除 : `fc-cache -fv`
3. パスを確認 : `fc-list | grep -i ipa`

得られたパスを`font_path`に記述して実行
これでも読み込まれない場合は、matplotlibのキャッシュを削除する

4. `rm ~/.cache/matplotlib/fontlist-v390.json` # 実際のファイル名に変更
"""
font_path: str = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
font_prop: FontProperties = FontProperties(fname=font_path)

# rcParamsでの全体的な設定
plt.rcParams.update(
    {
        # "font.family": ["Dejavu Sans"],
        "font.family": ["Dejavu Sans", font_prop.get_name()],
        "font.size": 30,
        "axes.labelsize": 30,
        "axes.titlesize": 30,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "legend.fontsize": 30,
    }
)

tag: str = "06_10-average-10_16"

project_home_dir: str = "/home/connect0459/labo/omu-eddy-covariance/workspace/seminar"

if __name__ == "__main__":
    plot_stacked_flux(
        input_filepath=f"{project_home_dir}/private/analyze_monthly-2024.12.18.csv",
        output_dir=f"{project_home_dir}/private",
        output_basename="ch4_flux_stacked_bar_directions-1218",
        tag=tag,
        ylim=100,
        # ylim=1.5,
        # concentration_unit="micro",
    )
