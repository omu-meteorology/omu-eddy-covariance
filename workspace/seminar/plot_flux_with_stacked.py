import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

"""
Ubuntu環境でのフォントの手動設定
ここでは日本語フォントを読み込んでいる
"""
font_path = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
font_prop = FontProperties(fname=font_path)

# rcParamsでの全体的な設定
plt.rcParams.update(
    {
        "font.family": ["Dejavu Sans", font_prop.get_name()],
        "font.size": 30,
        "axes.labelsize": 30,
        "axes.titlesize": 30,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "legend.fontsize": 30,
    }
)

project_home_dir: str = "/home/z23641k/labo/omu-eddy-covariance/workspace/seminar"

# データの読み込み
df = pd.read_csv(f"{project_home_dir}/private/analyze_monthly-for_graphs.csv")

# 方角の配置順序を定義（左上から時計回り）
directions_order = ["nw", "ne", "sw", "se"]
titles = {"nw": "北西", "ne": "北東", "sw": "南西", "se": "南東"}

# サブプロットを含む大きな図を作成
fig = plt.figure(figsize=(20, 13))

# 各方角についてサブプロットを作成
for idx, direction in enumerate(directions_order, 1):
    # サブプロットの位置を設定
    ax = fig.add_subplot(2, 2, idx)

    # 文字列を数値に変換
    diurnal = pd.to_numeric(df[f"diurnal_{direction}"], errors="coerce")
    gasratio = pd.to_numeric(df[f"gasratio_{direction}"], errors="coerce")

    # gas由来とbio由来のCH4フラックスを計算
    gas = diurnal * gasratio / 100
    bio = diurnal * (100 - gasratio) / 100

    # 積み上げ棒グラフの作成
    width = 0.8
    p1 = ax.bar(df["month"], gas, width, label="gas", color="orange")
    p2 = ax.bar(df["month"], bio, width, bottom=gas, label="bio", color="lightblue")

    # y軸の上限を設定
    # total_flux = gas + bio
    ax.set_ylim(0, 62)

    # gas比率の表示
    for i, (g, b) in enumerate(zip(gas, bio)):
        total = g + b
        ratio = g / total * 100
        ax.text(df["month"][i], total, f"{ratio:.0f}%", ha="center", va="bottom")

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
    "CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)",
    va="center",
    rotation="vertical",
)

# グラフの保存
plt.savefig(
    f"{project_home_dir}/private/ch4_flux_stacked_bar_directions.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()
