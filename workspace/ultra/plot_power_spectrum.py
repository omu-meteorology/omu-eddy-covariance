import pandas as pd
import matplotlib.pyplot as plt
from omu_eddy_covariance.ultra.eddydata_preprocessor import EddyDataPreprocessor
from omu_eddy_covariance.ultra.spectrum_calculator import SpectrumCalculator


def plot_power_spectrum(
    calculator: SpectrumCalculator,
    key: str,
    ylabel: str,
    color: str = "black",
    frequency_weighted: bool = True,
) -> None:
    """
    パワースペクトルを散布図としてプロットする関数

    Args:
        calculator (SpectrumCalculator): スペクトル計算用のクラスインスタンス
        key (str): プロットするデータの列名
        ylabel (str): ラベル
        frequency_weighted (bool, optional): 周波数の重みづけを適用するかどうか。デフォルトはTrue
    """
    plt.figure(figsize=(8, 6))

    # パワースペクトルの計算
    freqs, power_spectrum = calculator.calculate_powerspectrum(
        key=key,
        frequency_weighted=frequency_weighted,
        smooth=True,
    )

    # 散布図のプロット
    # plt.scatter(freqs, power_spectrum, c="black", s=50, alpha=0.6, label=key)
    plt.scatter(freqs, power_spectrum, c=color, s=100, label=key)

    # 軸を対数スケールに設定
    plt.xscale("log")
    plt.yscale("log")

    # -2/3 勾配の参照線
    plt.plot([0.01, 10], [10, 0.1], "-", color="black")
    plt.text(0.1, 0.1, "-2/3", fontsize=18)

    # plt.xlim(10**(-4),10**2)

    # ラベルとタイトルの設定（フォントサイズを18に設定）
    # plt.xlabel("f (Hz)", fontsize=18)
    # plt.ylabel(ylabel, fontsize=18)

    # # 目盛りのフォントサイズも調整
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    plt.xlabel("f (Hz)")
    plt.ylabel(ylabel)

    # 目盛りのフォントサイズも調整
    plt.xticks()
    plt.yticks()

    # グリッド線を描画
    # plt.grid(color="gray", linestyle="--")

    # グラフの表示
    plt.tight_layout()
    plt.show()


font_size_label: int = 20
font_size_ticks: int = 14
plt.rcParams["axes.labelsize"] = font_size_label
plt.rcParams["xtick.labelsize"] = font_size_ticks
plt.rcParams["ytick.labelsize"] = font_size_ticks
# 日本語フォントの設定
plt.rcParams["font.family"] = "MS Gothic"

# 使用例
data_dir = r"C:\Users\nakao\workspace\sac\ultra\data"
path = (
    data_dir
    # + r"\2024.06.21\Ultra_Eddy\eddy_csv-20240514_20240621\TOA5_37477.SAC_Ultra.Eddy_3_2024_06_01_0900.dat"
    + r"\2024.06.21\Ultra_Eddy\eddy_csv-20240514_20240621\TOA5_37477.SAC_Ultra.Eddy_3_2024_06_01_1500.dat"  # きれい
)
# path = r"C:\Users\nakao\workspace\sac\tmp\20240601-10_14.dat"
pre: EddyDataPreprocessor = EddyDataPreprocessor(path)
df: pd.DataFrame = pre.get_resampled_df()
calculator = SpectrumCalculator(
    df=df,
    fs=100,  # サンプリング周波数を適切に設定
    lag_index=-92,
    apply_lag_keys=[],
    plots=30,  # 40点の散布図
    apply_window=True,
)
plot_power_spectrum(
    calculator,
    key="Tv",
    ylabel=r"$fS_{\mathrm{Tv}} / s_{\mathrm{Tv}}^2$",
)
plot_power_spectrum(
    calculator,
    key="Ultra_CH4_ppm_C",
    ylabel=r"$fS_{\mathrm{CH_4}} / s_{\mathrm{CH_4}}^2$",
    color="red",
)
plot_power_spectrum(
    calculator,
    key="Ultra_C2H6_ppb",
    ylabel=r"$fS_{\mathrm{C_2H_6}} / s_{\mathrm{C_2H_6}}^2$",
    color="orange",
)
