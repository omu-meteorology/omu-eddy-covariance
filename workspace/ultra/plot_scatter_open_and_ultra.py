import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from matplotlib import rcParams
from datetime import time, timedelta, datetime
from matplotlib.ticker import FuncFormatter, MultipleLocator

# matplotlibの設定
font_size_label: int = 20
font_size_ticks: int = 16
rcParams["font.family"] = "Arial"
rcParams["font.size"] = font_size_label
rcParams["xtick.labelsize"] = font_size_ticks
rcParams["ytick.labelsize"] = font_size_ticks


def save_figure(fig, filename):
    """図をoutputsディレクトリに保存する"""
    out_dir: str = "C:\\Users\\nakao\\workspace\\sac\\seminar\\outputs"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_diurnal_patterns(
    df,
    x_col,
    y_cols_ch4,
    y_cols_c2h6,
    labels_ch4,
    labels_c2h6,
    colors_ch4,
    colors_c2h6,
    filename,
):
    """CH4とC2H6の日変化パターンを1つの図に並べてプロットする"""
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # CH4のプロット (左側)
    for y_col, label, color in zip(y_cols_ch4, labels_ch4, colors_ch4):
        ax1.plot(df[x_col], df[y_col], "-o", label=label, color=color)

    ax1.set_xlabel("Time")
    ax1.set_ylabel(r"CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)")
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax1.set_xlim(df[x_col].min(), df[x_col].max())
    ax1.set_ylim(0, 100)
    ax1.yaxis.set_major_locator(MultipleLocator(20))
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.0f}"))
    ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, va="top")

    # C2H6のプロット (右側)
    for y_col, label, color in zip(y_cols_c2h6, labels_c2h6, colors_c2h6):
        ax2.plot(df[x_col], df[y_col], "-o", label=label, color=color)

    ax2.set_xlabel("Time")
    ax2.set_ylabel(r"C$_2$H$_6$ Flux (nmol m$^{-2}$ s$^{-1}$)")
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax2.set_xlim(df[x_col].min(), df[x_col].max())
    ax2.set_ylim(0, 5.0)
    ax2.yaxis.set_major_locator(MultipleLocator(1))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))
    ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, va="top")

    plt.tight_layout()
    save_figure(fig, filename)


def plot_scatter_with_regression(df, x_col, y_col, xlabel, ylabel, filename):
    """散布図と回帰直線をプロットする"""
    numeric_df = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(numeric_df[x_col], numeric_df[y_col], color="black")

    # 線形回帰
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        numeric_df[x_col], numeric_df[y_col]
    )

    # 近似直線
    x_range = np.linspace(0, 80, 100)
    y_range = slope * x_range + intercept
    ax.plot(x_range, y_range, "r")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # 近似直線の式を右上に赤色で表示（符号を調整）
    equation = f"y = {slope:.2f}x {'+' if intercept >= 0 else '-'} {abs(intercept):.2f}"
    ax.text(
        0.95, 0.95, equation, transform=ax.transAxes, va="top", ha="right", color="red"
    )

    # 決定係数を近似直線の式の下に表示
    ax.text(
        0.95,
        0.88,
        f"R² = {r_value**2:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        color="red",
    )

    save_figure(fig, filename)


if __name__ == "__main__":
    # CSVファイルの読み込み
    csv_path: str = "C:\\Users\\nakao\\workspace\\sac\\seminar\\data\\diurnals.csv"
    df: pd.DataFrame = pd.read_csv(csv_path)
    df = df.iloc[1:]  # 最初の行をスキップ

    # 数値データを含む列を数値型に変換
    numeric_columns = ["Fch4_open", "Fch4_ultra", "Fc2h6_ultra"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    # 時間を24時間形式に変換
    df["Time"] = pd.to_datetime(df["Hour"], format="%H:%M", errors="coerce").dt.time

    # 無効な時間データを除外
    df = df.dropna(subset=["Time"])

    # 24時（0時）のデータを追加して24時間のサイクルを完成させる
    last_row = df.iloc[0].copy()
    last_row["Time"] = time(0, 0)  # 24:00 の代わりに 00:00 を使用
    df = pd.concat([df, pd.DataFrame([last_row])], ignore_index=True)

    # 時間をdatetime型に変換（日付は任意）
    base_date = datetime(2023, 1, 1)
    df["Datetime"] = pd.to_datetime(
        base_date.strftime("%Y-%m-%d ") + df["Time"].astype(str)
    )

    # 最後の行（00:00）を次の日の00:00として処理
    last_datetime = df.loc[df.index[-1], "Datetime"]

    if pd.isna(last_datetime):
        print("警告: 最後の行のDatetimeがNaNです。スキップします。")
    elif isinstance(last_datetime, (pd.Timestamp, datetime, np.datetime64)):
        df.loc[df.index[-1], "Datetime"] = pd.to_datetime(last_datetime) + timedelta(
            days=1
        )
    elif isinstance(last_datetime, str):
        df.loc[df.index[-1], "Datetime"] = pd.to_datetime(last_datetime) + timedelta(
            days=1
        )
    else:
        print(
            f"警告: 予期しない型 {type(last_datetime)} です。最後の行の処理をスキップします。"
        )

    # print(df["Datetime"].dtype)  # 確認のため、Datetime列の型を出力

    # 1. CH4とC2H6の日変化パターンを1つの図にプロット
    plot_combined_diurnal_patterns(
        df,
        "Datetime",
        ["Fch4_ultra", "Fch4_open"],
        ["Fc2h6_ultra"],
        ["Ultra", "Open Path"],
        ["Ultra"],
        ["black", "red"],
        ["black"],
        "combined_diurnal_patterns.png",
    )

    # 2. OpenのCH4フラックスとUltraのCH4フラックスの散布図
    plot_scatter_with_regression(
        df,
        "Fch4_open",
        "Fch4_ultra",
        r"Open Path CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
        r"Ultra CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
        "ch4_flux_comparison.png",
    )

    print("グラフが指定されたディレクトリに作成されました。")
