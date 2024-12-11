import os
import matplotlib.pyplot as plt
from omu_eddy_covariance import EddyDataPreprocessor, SpectrumCalculator


def plot_power_spectrum(
    calculator: SpectrumCalculator,
    key: str,
    ylabel: str,
    frequency_weighted: bool = True,
    output_dir: str | None = None,
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
    freqs, power_spectrum = calculator.calculate_power_spectrum(
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
    if output_dir is not None:
        plt.savefig(
            f"{output_dir}/power-{key}",
            dpi=300,
            bbox_inches="tight",
        )


# 変数定義
input_file_path = "/home/connect0459/labo/omu-eddy-covariance/workspace/ultra/private/data/2024.10.10/eddy_csv/TOA5_37477.SAC_Ultra.Eddy_107_2024_10_10_1200.dat"
output_dir: str = (
    "/home/connect0459/labo/omu-eddy-covariance/workspace/ultra/private/outputs"
)
font_size_label: int = 20
font_size_ticks: int = 14

plt.rcParams["axes.labelsize"] = font_size_label
plt.rcParams["xtick.labelsize"] = font_size_ticks
plt.rcParams["ytick.labelsize"] = font_size_ticks
# 日本語フォントの設定
plt.rcParams["font.family"] = ["Arial", "Dejavu Sans"]

if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    edp: EddyDataPreprocessor = EddyDataPreprocessor()
    df, _ = edp.get_resampled_df(filepath=input_file_path)
    calculator = SpectrumCalculator(
        df=df,
        fs=10,
        apply_lag_keys=["Ultra_CH4_ppm_C", "Ultra_C2H6_ppb"],
        lag_second=11.95,
    )
    plot_power_spectrum(
        calculator,
        key="Tv",
        output_dir=output_dir,
        ylabel=r"$fS_{\mathrm{Tv}} / s_{\mathrm{Tv}}^2$",
    )
    plot_power_spectrum(
        calculator,
        key="Ultra_CH4_ppm_C",
        output_dir=output_dir,
        plot_color="red",
        ylabel=r"$fS_{\mathrm{CH_4}} / s_{\mathrm{CH_4}}^2$",
    )
    plot_power_spectrum(
        calculator,
        key="Ultra_C2H6_ppb",
        output_dir=output_dir,
        plot_color="orange",
        ylabel=r"$fS_{\mathrm{C_2H_6}} / s_{\mathrm{C_2H_6}}^2$",
    )
