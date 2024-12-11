import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def calculate_power_spectrum(data_path):
    # CSVファイルの読み込み（ヘッダーの処理）
    df = pd.read_csv(data_path, skiprows=[0, 2, 3])

    # CH4データの取得
    # key:str = "Ultra_CH4_ppm_C"
    key:str = "Ultra_C2H6_ppb"
    ch4_data = df[key].values

    # サンプリング周波数の計算（10Hzを想定）
    fs = 10.0

    # トレンド除去
    ch4_detrended = signal.detrend(ch4_data)

    # ハミング窓の適用
    window = signal.windows.hamming(len(ch4_detrended))
    ch4_windowed = ch4_detrended * window

    # パワースペクトルの計算
    frequencies, psd = signal.welch(
        ch4_windowed, fs=fs, window="hamming", nperseg=1024, scaling="density"
    )

    # 周波数による重み付け
    weighted_psd = frequencies * psd

    # 対数スケールで等間隔な30点を生成
    log_freq_min = np.log10(0.001)  # 固定の最小周波数
    log_freq_max = np.log10(frequencies[-1])
    log_freq_resampled = np.logspace(log_freq_min, log_freq_max, 30)

    # 最近傍補間で対応する重み付けパワースペクトル値を取得
    weighted_psd_resampled = np.interp(
        log_freq_resampled, frequencies, weighted_psd, left=np.nan, right=np.nan
    )

    # プロット作成（3つのサブプロット）
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # 通常のパワースペクトル
    ax1.loglog(frequencies, psd)
    ax1.grid(True)
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Power Spectral Density [(ppm)²/Hz]")
    ax1.set_title("CH4 Power Spectrum")
    ax1.set_xlim([0.001, fs / 2])

    # 重み付けしたパワースペクトル
    ax2.loglog(frequencies, weighted_psd)
    ax2.grid(True)
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Weighted Power Spectral Density [f*(ppm)²/Hz]")
    ax2.set_title("Frequency-Weighted CH4 Power Spectrum")
    ax2.set_xlim([0.001, fs / 2])

    # リサンプリングした重み付けパワースペクトル
    valid_mask = ~np.isnan(weighted_psd_resampled)
    ax3.loglog(log_freq_resampled[valid_mask], weighted_psd_resampled[valid_mask], "o-")
    ax3.grid(True)
    ax3.set_xlabel("Frequency [Hz]")
    ax3.set_ylabel("Weighted Power Spectral Density [f*(ppm)²/Hz]")
    ax3.set_title("Resampled Frequency-Weighted CH4 Power Spectrum (30 points)")
    ax3.set_xlim([0.001, fs / 2])

    # プロットの体裁を整える
    plt.tight_layout()

    plt.savefig(
        f"/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/outputs/tests/test-power-{key}.png"
    )

    # プロットを閉じる
    plt.close()

    return (
        frequencies,
        psd,
        weighted_psd,
        log_freq_resampled[valid_mask],
        weighted_psd_resampled[valid_mask],
    )


# 使用例
frequencies, psd, weighted_psd, log_freq_resampled, weighted_psd_resampled = (
    calculate_power_spectrum(
        "/home/connect0459/labo/omu-eddy-covariance/workspace/senior_thesis/private/data/eddy_csv-resampled-06/TOA5_37477.SAC_Ultra.Eddy_1_2024_06_21_1500-resampled.csv"
    )
)
