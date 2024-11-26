import pandas as pd
import matplotlib.pyplot as plt
from omu_eddy_covariance import EddyDataPreprocessor
from omu_eddy_covariance import SpectrumCalculator


def plot_power_spectrum(
    calculator: SpectrumCalculator,
    key: str,
    ylabel: str,
    frequency_weighted: bool = True,
    plot_color: str = "black",
) -> None:
    """
    パワースペクトルを散布図としてプロットする関数

    Args:
        calculator (SpectrumCalculator): スペクトル計算用のクラスインスタンス
        key (str): プロットするデータの列名
        ylabel (str): y軸のラベル
        frequency_weighted (bool, optional): 周波数の重みづけを適用するかどうか。デフォルトはTrue
        plot_color (str): プロット点の色。デフォルトは'black'。
    """
    plt.figure(figsize=(10, 8))

    # パワースペクトルの計算
    freqs, power_spectrum = calculator.calculate_powerspectrum(
        key=key,
        frequency_weighted=frequency_weighted,
        smooth=True,
    )

    # 散布図のプロット
    plt.scatter(freqs, power_spectrum, c=plot_color, s=100, label=key)

    # 軸を対数スケールに設定
    plt.xscale("log")
    plt.yscale("log")

    # -2/3 勾配の参照線
    plt.plot([0.01, 10], [1, 0.01], "-", color="black")
    plt.text(0.1, 0.1, "-2/3", fontsize=18)

    # # 目盛りのフォントサイズも調整
    plt.xlabel("f (Hz)")
    plt.ylabel(ylabel)

    # 目盛りのフォントサイズも調整
    plt.xticks()
    plt.yticks()

    # グラフの表示
    plt.tight_layout()
    plt.savefig(
        f"/home/z23641k/labo/omu-eddy-covariance/workspace/ultra/private/data/output/power-{key}",
        dpi=300,
        bbox_inches="tight",
    )


font_size_label: int = 20
font_size_ticks: int = 14
plt.rcParams["axes.labelsize"] = font_size_label
plt.rcParams["xtick.labelsize"] = font_size_ticks
plt.rcParams["ytick.labelsize"] = font_size_ticks
# 日本語フォントの設定
plt.rcParams["font.family"] = ["Arial", "Dejavu Sans"]

if __name__ == "__main__":
    # 使用例
    data_dir = r"C:\Users\nakao\workspace\sac\ultra\data"
    # path = (
    #     data_dir
    #     # + r"\2024.06.21\Ultra_Eddy\eddy_csv-20240514_20240621\TOA5_37477.SAC_Ultra.Eddy_3_2024_06_01_0900.dat"
    #     + r"\2024.06.21\Ultra_Eddy\eddy_csv-20240514_20240621\TOA5_37477.SAC_Ultra.Eddy_3_2024_06_01_1500.dat"  # きれい
    # )
    path = "/home/z23641k/labo/omu-eddy-covariance/workspace/ultra/private/data/eddy_csv/TOA5_37477.SAC_Ultra.Eddy_107_2024_10_10_1200.dat"

    edp: EddyDataPreprocessor = EddyDataPreprocessor()
    df: pd.DataFrame = edp.get_resampled_df(filepath=path)
    calculator = SpectrumCalculator(
        df=df,
        fs=10,
        apply_delay_keys=["Ultra_CH4_ppm_C", "Ultra_C2H6_ppb"],
        delay_second=11.95,
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
        plot_color="red",
    )
    plot_power_spectrum(
        calculator,
        key="Ultra_C2H6_ppb",
        ylabel=r"$fS_{\mathrm{C_2H_6}} / s_{\mathrm{C_2H_6}}^2$",
        plot_color="orange",
    )
