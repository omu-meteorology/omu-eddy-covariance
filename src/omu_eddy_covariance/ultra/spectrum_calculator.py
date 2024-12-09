import numpy as np
import pandas as pd
from scipy import signal


class SpectrumCalculator:
    def __init__(
        self,
        df: pd.DataFrame,
        apply_lag_keys: list[str],
        lag_second: float,
        fs: float,
        apply_window: bool = True,
        dimensionless: bool = True,
        plots: int = 30,
    ):
        """
        データロガーから取得したデータファイルを用いて計算を行うクラス。

        Args:
            df (pd.DataFrame): pandasのデータフレーム。解析対象のデータを含む。
            apply_lag_keys (list[str]): コスペクトルの遅れ時間補正を適用するキーのリスト。
            lag_second (float): 遅延時間（秒）。データの遅延を指定。
            fs (float): サンプリング周波数（Hz）。データのサンプリングレートを指定。
            apply_window (bool, optional): 窓関数を適用するフラグ。デフォルトはTrue。
            dimensionless (bool, optional): Trueの場合、分散で割って無次元化を行う。デフォルトはTrue。
            plots (int): プロットする点の数。可視化のためのデータポイント数。
        """
        self.apply_lag_keys: list[str] = apply_lag_keys
        self.apply_window: bool = apply_window
        self.lag_second: float = lag_second
        self.dimensionless: bool = dimensionless
        self.df: pd.DataFrame = df
        self.fs: float = fs
        self.plots: int = plots
        self.window_type: str = "hamming"

    def calculate_cospectrum(
        self,
        key1: str,
        key2: str,
        frequency_weighted: bool = True,
        interpolate_points: bool = True,
        smooth: bool = False,
    ) -> tuple:
        """
        DataFrameから指定されたkey1とkey2のコスペクトルを計算する
        fft.cと同様のロジックで実装

        Args:
            key1 (str): データの列名1
            key2 (str): データの列名2
            frequency_weighted (bool, optional): 周波数の重みづけを適用するかどうか。デフォルトはTrue。
            interpolate_points (bool, optional): 等間隔なデータ点を生成するかどうか（対数軸上で等間隔）
            smooth (bool, optional): スペクトルを平滑化するかどうか。デフォルトはFalse。

        Returns:
            tuple: (freqs, cospectrum, correlation_coefficient)
                - freqs (np.ndarray): 周波数軸（対数スケールの場合は対数変換済み）
                - cospectrum (np.ndarray): コスペクトル（対数スケールの場合は対数変換済み）
                - correlation_coefficient (float): 変数の相関係数
        """
        freqs, cospectrum, _, correlation_coefficient = self.calculate_crossspectrum(
            key1=key1,
            key2=key2,
            frequency_weighted=frequency_weighted,
            interpolate_points=interpolate_points,
            smooth=smooth,
        )
        return freqs, cospectrum, correlation_coefficient

    def calculate_crossspectrum(
        self,
        key1: str,
        key2: str,
        frequency_weighted: bool = True,
        interpolate_points: bool = True,
        smooth: bool = False,
    ) -> tuple:
        """
        DataFrameから指定されたkey1とkey2のコスペクトルとクアドラチャスペクトルを計算する
        fft.cと同様のロジックで実装

        Args:
            key1 (str): データの列名1
            key2 (str): データの列名2
            frequency_weighted (bool, optional): 周波数の重みづけを適用するかどうか。デフォルトはTrue。
            interpolate_points (bool, optional): 等間隔なデータ点を生成するかどうか（対数軸上で等間隔）
            smooth (bool, optional): スペクトルを平滑化するかどうか。デフォルトはFalse。

        Returns:
            tuple: (freqs, cospectrum, quadrature_spectrum, correlation_coefficient)
                - freqs (np.ndarray): 周波数軸（対数スケールの場合は対数変換済み）
                - cospectrum (np.ndarray): コスペクトル（対数スケールの場合は対数変換済み）
                - quadrature_spectrum (np.ndarray): クアドラチャスペクトル（対数スケールの場合は対数変換済み）
                - correlation_coefficient (float): 変数の相関係数
        """
        # key1とkey2に一致するデータを取得
        data1: np.ndarray = np.array(self.df[key1].values)
        data2: np.ndarray = np.array(self.df[key2].values)

        # 遅れ時間の補正
        if key2 in self.apply_lag_keys:
            data1, data2 = SpectrumCalculator._correct_lag_time(
                data1=data1, data2=data2, fs=self.fs, lag_second=self.lag_second
            )

        # トレンド除去
        data1 = SpectrumCalculator._detrend(data=data1, fs=self.fs, first=True)
        data2 = SpectrumCalculator._detrend(data=data2, fs=self.fs, first=True)

        # データ長
        data_length: int = len(data1)

        # 共分散の計算
        cov_matrix: np.ndarray = np.cov(data1, data2)
        covariance: float = cov_matrix[0, 1]

        # 窓関数の適用
        window_scale = 1.0
        if self.apply_window:
            window = SpectrumCalculator._generate_window_function(
                type=self.window_type, data_length=data_length
            )
            data1 *= window
            data2 *= window
            window_scale = np.mean(window**2)

        # FFTの計算
        fft1 = np.fft.rfft(data1)
        fft2 = np.fft.rfft(data2)

        # 周波数軸の作成
        freqs: np.ndarray = np.fft.rfftfreq(data_length, 1.0 / self.fs)

        # fft.cと同様のコスペクトル計算ロジック
        cospectrum = np.zeros(len(freqs))
        quad_spectrum = np.zeros(len(freqs))

        for i in range(1, len(freqs)):  # 0Hz成分を除外
            z1 = fft1[i]
            z2 = fft2[i]
            z1_star = np.conj(z1)
            z2_star = np.conj(z2)

            # x1 = z1 + z1*, x2 = z2 + z2*
            x1 = z1 + z1_star
            x2 = z2 + z2_star
            x1_re = x1.real
            x1_im = x1.imag
            x2_re = x2.real
            x2_im = x2.imag

            # y1 = z1 - z1*, y2 = z2 - z2*
            y1 = z1 - z1_star
            y2 = z2 - z2_star
            # 虚部と実部を入れ替え
            y1_re = y1.imag
            y1_im = -y1.real
            y2_re = y2.imag
            y2_im = -y2.real

            # コスペクトルとクァドラチャスペクトルの計算
            conj_x1_x2 = complex(
                x1_re * x2_re + x1_im * x2_im, x1_im * x2_re - x1_re * x2_im
            )
            conj_y1_y2 = complex(
                y1_re * y2_re + y1_im * y2_im, y1_im * y2_re - y1_re * y2_im
            )

            # スケーリングを適用
            cospectrum[i] = (conj_x1_x2.real) * 0.5 / (data_length * window_scale)
            quad_spectrum[i] = (conj_y1_y2.real) * 0.5 / (data_length * window_scale)

        # 無次元化
        if self.dimensionless:
            cospectrum /= covariance
            quad_spectrum /= covariance

        # 周波数の重みづけ
        if frequency_weighted:
            cospectrum[1:] *= freqs[1:]
            quad_spectrum[1:] *= freqs[1:]

        # 3点移動平均によるスペクトルの平滑化
        if smooth:
            smoothed_co = np.zeros_like(cospectrum)
            smoothed_quad = np.zeros_like(quad_spectrum)

            # 端点の処理
            smoothed_co[0] = 0.5 * (cospectrum[0] + cospectrum[1])
            smoothed_co[-1] = 0.5 * (cospectrum[-2] + cospectrum[-1])
            smoothed_quad[0] = 0.5 * (quad_spectrum[0] + quad_spectrum[1])
            smoothed_quad[-1] = 0.5 * (quad_spectrum[-2] + quad_spectrum[-1])

            # 中間点の平滑化
            for i in range(1, len(cospectrum) - 1):
                smoothed_co[i] = (
                    0.25 * cospectrum[i - 1]
                    + 0.5 * cospectrum[i]
                    + 0.25 * cospectrum[i + 1]
                )
                smoothed_quad[i] = (
                    0.25 * quad_spectrum[i - 1]
                    + 0.5 * quad_spectrum[i]
                    + 0.25 * quad_spectrum[i + 1]
                )

            cospectrum = smoothed_co
            quad_spectrum = smoothed_quad

        # 相関係数の計算
        correlation_coefficient: float = np.corrcoef(data1, data2)[0, 1]

        # 0Hz成分を除外
        nonzero_mask = freqs != 0
        freqs = freqs[nonzero_mask]
        cospectrum = cospectrum[nonzero_mask]
        quad_spectrum = quad_spectrum[nonzero_mask]

        if interpolate_points:
            # 対数変換（スペクトルは負の値を取りうるので絶対値を取る）
            log_freqs = np.log10(freqs)
            log_co = np.log10(np.abs(cospectrum))
            log_quad = np.log10(np.abs(quad_spectrum))

            # 周波数軸の最小値と最大値を取得
            min_freq = np.min(log_freqs)
            max_freq = np.max(log_freqs)

            # 等間隔なplots個の点を生成（対数軸上で等間隔）
            interp_log_freqs = np.linspace(min_freq, max_freq, self.plots)
            interp_freqs = 10**interp_log_freqs

            # 生成した周波数に対応するスペクトルの値を対数軸上で線形補間
            interp_log_co = np.interp(interp_log_freqs, log_freqs, log_co)
            interp_log_quad = np.interp(interp_log_freqs, log_freqs, log_quad)
            interp_cospectrum = 10**interp_log_co
            interp_quadrature_spectrum = 10**interp_log_quad

            return (
                interp_freqs,
                interp_cospectrum,
                interp_quadrature_spectrum,
                correlation_coefficient,
            )
        else:
            # 線形スケールの場合はそのまま返す
            return freqs, cospectrum, quad_spectrum, correlation_coefficient

    def calculate_power_spectrum_density(
        self,
        key: str,
        dimensionless: bool = True,
        frequency_weighted: bool = True,
        interpolate_points: bool = True,
        smooth: bool = True,
    ) -> tuple:
        """
        DataFrameから指定されたkeyのパワースペクトルと周波数軸を計算する
        scipy.signal.welchを使用してパワースペクトルを計算

        Args:
            key (str): データの列名
            frequency_weighted (bool, optional): 周波数の重みづけを適用するかどうか。デフォルトはTrue。
            interpolate_points (bool, optional): 等間隔なデータ点を生成するかどうか（対数軸上で等間隔）
            smooth (bool, optional): パワースペクトルを平滑化するかどうか。デフォルトはFalse。

        Returns:
            tuple: (freqs, power_spectrum)
                - freqs (np.ndarray): 周波数軸（対数スケールの場合は対数変換済み）
                - power_spectrum (np.ndarray): パワースペクトル（対数スケールの場合は対数変換済み）
        """
        # データの取得とトレンド除去
        data = np.array(self.df[key].values)
        data = SpectrumCalculator._detrend(data, self.fs)

        # welchメソッドでパワースペクトル計算
        freqs, psd = signal.welch(
            data, fs=self.fs, window=self.window_type, nperseg=1024, scaling="density"
        )

        # 周波数の重みづけ（0Hz除外の前に実施）
        if frequency_weighted:
            psd = freqs * psd

        # 無次元化（0Hz除外の前に実施）
        if dimensionless:
            variance = np.var(data)
            psd /= variance

        # スムージング（0Hz除外の前に実施）
        if smooth:
            psd = self._smooth_spectrum(psd)

        if interpolate_points:
            # 補間処理（0Hz除外の前に実施）
            log_freq_min = np.log10(0.001)
            log_freq_max = np.log10(freqs[-1])
            log_freq_resampled = np.logspace(log_freq_min, log_freq_max, 30)

            psd_resampled = np.interp(
                log_freq_resampled, freqs, psd, left=np.nan, right=np.nan
            )

            # NaNを除外
            valid_mask = ~np.isnan(psd_resampled)
            freqs = log_freq_resampled[valid_mask]
            psd = psd_resampled[valid_mask]

        # 0Hz成分を最後に除外
        nonzero_mask = freqs != 0
        freqs = freqs[nonzero_mask]
        psd = psd[nonzero_mask]

        return freqs, psd

    @staticmethod
    def _correct_lag_time(
        data1: np.ndarray,
        data2: np.ndarray,
        fs: float,
        lag_second: float,
    ) -> tuple:
        """
        相互相関関数を用いて遅れ時間を補正する
        コスペクトル計算に使用

        Args:
            data1 (np.ndarray): 基準データ
            data2 (np.ndarray): 遅れているデータ
            fs (float): サンプリング周波数
            lag_second (float): data1からdata2が遅れている時間（秒）。負の値は許可されない。

        Returns:
            tuple: (data1, data2)
                - data1 (np.ndarray): 補正された基準データ
                - data2 (np.ndarray): 補正された遅れているデータ

        Raises:
            ValueError: lag_secondが負の値の場合
        """
        if lag_second < 0:
            raise ValueError("lag_second must be non-negative.")
        # lag_secondをサンプリング周波数でスケーリングしてインデックスに変換
        lag_index: int = int(lag_second * fs)
        # データ1とデータ2の共通部分を抽出
        data1 = data1[lag_index:]
        data2 = data2[:-lag_index]
        return data1, data2

    @staticmethod
    def _detrend(
        data: np.ndarray, fs: float, first: bool = True, second: bool = False
    ) -> np.ndarray:
        """
        データから一次トレンドおよび二次トレンドを除去します。

        Args:
            data (np.ndarray): 入力データ
            fs (float): サンプリング周波数
            first (bool, optional): 一次トレンドを除去するかどうか. デフォルトはTrue.
            second (bool, optional): 二次トレンドを除去するかどうか. デフォルトはFalse.

        Returns:
            np.ndarray: トレンド除去後のデータ

        Raises:
            ValueError: first と second の両方がFalseの場合
        """

        if not (first or second):
            raise ValueError("少なくとも一次または二次トレンドの除去を指定してください")

        detrended_data: np.ndarray = data.copy()

        # 一次トレンドの除去
        if first:
            detrended_data = signal.detrend(detrended_data)

        # 二次トレンドの除去
        if second:
            # 二次トレンドを除去するために、まず一次トレンドを除去
            detrended_data = signal.detrend(detrended_data, type="linear")
            # 二次トレンドを除去するために、二次多項式フィッティングを行う
            coeffs_second = np.polyfit(
                np.arange(len(detrended_data)), detrended_data, 2
            )
            trend_second = np.polyval(coeffs_second, np.arange(len(detrended_data)))
            detrended_data = detrended_data - trend_second

        return detrended_data

    @staticmethod
    def _generate_window_function(type: str, data_length: int) -> np.ndarray:
        """
        指定された種類の窓関数を適用する

        Args:
            type (str): 窓関数の種類 ('hanning', 'hamming', 'blackman')
            data_length (int): データ長

        Returns:
            np.ndarray: 適用された窓関数

        Notes:
            - 指定された種類の窓関数を適用し、numpy配列として返す
            - 無効な種類が指定された場合、警告を表示しHann窓を適用する
        """
        if type == "hanning":
            return np.hanning(data_length)
        elif type == "hamming":
            return np.hamming(data_length)
        elif type == "blackman":
            return np.blackman(data_length)
        else:
            print('Warning: Invalid argument "type". Return hanning window.')
            return np.hanning(data_length)

    @staticmethod
    def _smooth_spectrum(spectrum: np.ndarray) -> np.ndarray:
        """
        3点移動平均によるスペクトルの平滑化を行う
        """
        smoothed = np.zeros_like(spectrum)

        # 端点の処理
        smoothed[0] = 0.5 * (spectrum[0] + spectrum[1])
        smoothed[-1] = 0.5 * (spectrum[-2] + spectrum[-1])

        # 中間点の平滑化
        for i in range(1, len(spectrum) - 1):
            smoothed[i] = (
                0.25 * spectrum[i - 1] + 0.5 * spectrum[i] + 0.25 * spectrum[i + 1]
            )

        return smoothed
