import numpy as np
import pandas as pd


class SpectrumCalculator:
    def __init__(
        self,
        df: pd.DataFrame,
        fs: float,
        lag_index: int,
        apply_lag_keys: list[str],
        plots: int,
        apply_window: bool,
        dimensionless: bool = True,
    ):
        """
        データロガーから取得したデータファイルを用いて計算を行うクラス。

        Args:
            df (pd.DataFrame): pandasのデータフレーム
            fs (float): サンプリング周波数（Hz）.
            lag_index (int): 相互相関のピークのインデックス
            apply_lag_keys (list[int]): コスペクトルの遅れ時間補正を適用するキー
            plots (int): プロットする点の数.
            apply_window (bool, optional): 窓関数を適用するフラグ. Defaults to True.
            dimensionless (bool, optional): Trueのとき分散で割って無次元化を行う. Defaults to True.
        """
        self.df: pd.DataFrame = df
        self.fs: float = fs
        self.lag_index: int = lag_index
        self.apply_lag_keys: list[str] = apply_lag_keys
        self.plots: int = plots
        self.apply_window: bool = apply_window
        self.window_type: str = "hamming"
        self.dimensionless: bool = dimensionless

    def __detrend(self, data: np.ndarray, deg: int) -> np.ndarray:
        """
        データの線形トレンドを除去する

        Args:
            data (np.ndarray): 入力データ

        Returns:
            np.ndarray: トレンド除去後のデータ
        """
        time = np.arange(len(data))
        coeffs = np.polyfit(time, data, deg)
        trend = np.polyval(coeffs, time)
        return data - trend

    def __generate_window_function(self, type: str, data_length: int) -> np.ndarray:
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

    def __correct_time_lag(
        self, lag_index: int, data1: np.ndarray, data2: np.ndarray
    ) -> tuple:
        """
        相互相関関数を用いて遅れ時間を補正する
        コスペクトル計算に使用

        Args:
            lag_index (int): 相互相関のピークのインデックス
            data1 (np.ndarray): データ1
            data2 (np.ndarray): データ2

        Returns:
            tuple: (data1, data2)
                - data1 (np.ndarray): 補正されたデータ1
                - data2 (np.ndarray): 補正されたデータ2
        """
        # データ1とデータ2の共通部分を抽出
        if lag_index > 0:
            data1 = data1[lag_index:]
            data2 = data2[:-lag_index]
        elif lag_index < 0:
            data1 = data1[:lag_index]
            data2 = data2[-lag_index:]
        return data1, data2

    def __moving_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """
        指定されたウィンドウサイズでデータの移動平均を計算します。

        Args:
            data (np.ndarray): 移動平均を計算する対象のデータ配列
            window_size (int): 移動平均を計算するためのウィンドウサイズ

        Returns:
            np.ndarray: 移動平均が適用されたデータ配列
        """
        window = np.ones(window_size) / window_size
        return np.convolve(data, window, mode="valid")

    def calculate_powerspectrum(
        self,
        key: str,
        frequency_weighted: bool = True,
        smooth: bool = False,
        window_size: int = 30,
    ) -> tuple:
        """
        DataFrameから指定されたkeyのパワースペクトルと周波数軸を計算する

        Args:
            key (str): データの列名
            frequency_weighted (bool, optional): 周波数の重みづけを適用するかどうか。デフォルトはTrue。
            smooth (bool, optional): パワースペクトルを平滑化するかどうか。デフォルトはFalse。
            window_size (int, optional): 平滑化のための移動平均ウィンドウサイズ。デフォルトは30。

        Returns:
            tuple: (interp_power_spectrum, interp_freqs)
                - interp_freqs (np.ndarray): 補間された周波数軸
                - interp_power_spectrum (np.ndarray): 補間されたパワースペクトル
        """
        # keyに一致するデータを取得
        column_data: np.ndarray = np.array(self.df[key].values)

        # データ長
        data_length: int = len(column_data)

        # トレンド除去
        column_data = self.__detrend(column_data, 1)

        # 窓関数適用前データの分散を計算
        variance = np.var(column_data)

        window: np.ndarray = np.array([])
        if self.apply_window:
            # ハニング窓を生成
            window = self.__generate_window_function(
                type=self.window_type, data_length=data_length
            )
            # ハニング窓を適用
            column_data *= window

        # パワースペクトルを計算
        power_spectrum: np.ndarray = np.abs(np.fft.rfft(column_data)) ** 2 / data_length

        # 窓関数の平均二乗値で割ることでスケーリングを行う
        if self.apply_window:
            power_spectrum /= np.mean(window**2)

        # 無次元化を行う
        if self.dimensionless:
            power_spectrum /= variance

        # 周波数軸
        rfft_freq: np.ndarray = np.fft.rfftfreq(data_length, 1.0 / self.fs)

        # 周波数が0でない要素のみを選択
        nonzero_mask: np.ndarray = rfft_freq != 0
        nonzero_freqs: np.ndarray = rfft_freq[nonzero_mask]
        nonzero_power_spectrum: np.ndarray = power_spectrum[nonzero_mask]

        # # 周波数の加法的重みづけ
        # if frequency_weighted:
        #     nonzero_power_spectrum *= nonzero_freqs  # 各周波数に対応する値をかける

        # 周波数軸とパワースペクトルの対数を取る
        log_freqs: np.ndarray = np.log10(nonzero_freqs)
        log_power_spectrum: np.ndarray = np.log10(nonzero_power_spectrum)

        # 周波数の乗法的重みづけ
        if frequency_weighted:
            log_power_spectrum *= log_freqs  # 各周波数に対応する値をかける

        if smooth:
            # データ端の処理のため、端点を複製
            padded_data = np.pad(
                log_power_spectrum, (window_size // 2, window_size // 2), mode="edge"
            )
            # 移動平均を適用
            log_power_spectrum = self.__moving_average(padded_data, window_size)
            # 元の長さにトリミング
            if window_size % 2 == 0:
                log_power_spectrum = log_power_spectrum[:-1]

        # 周波数軸の最小値と最大値を取得
        min_freq: float = np.min(log_freqs)
        max_freq: float = np.max(log_freqs)

        # 等間隔なplots個の点を生成（対数軸上で等間隔）
        interp_log_freqs: np.ndarray = np.linspace(min_freq, max_freq, self.plots)
        interp_freqs: np.ndarray = 10**interp_log_freqs

        # 生成した周波数に対応するパワースペクトルの値を対数軸上で線形補間
        interp_log_power_spectrum: np.ndarray = np.interp(
            interp_log_freqs, log_freqs, log_power_spectrum
        )
        interp_power_spectrum: np.ndarray = 10**interp_log_power_spectrum

        return interp_freqs, interp_power_spectrum

    def calculate_cospectrum(
        self,
        key1: str,
        key2: str,
        frequency_weighted: bool = True,
        smooth: bool = False,
        window_size: int = 30,
    ) -> tuple:
        """
        DataFrameから指定されたkey1とkey2のコスペクトルを計算する

        Args:
            key1 (str): データの列名1
            key2 (str): データの列名2
            frequency_weighted (bool, optional): 周波数の重みづけを適用するかどうか。デフォルトはTrue。
            smooth (bool, optional): パワースペクトルを平滑化するかどうか。デフォルトはFalse。
            window_size (int, optional): 平滑化のための移動平均ウィンドウサイズ。デフォルトは30。

        Returns:
            tuple: (interp_freqs, interp_cospectrum, correlation_coefficient)
                - interp_freqs (np.ndarray): 補間された周波数軸
                - interp_cospectrum (np.ndarray): 補間されたコスペクトル
                - correlation_coefficient (float): 変数の相関係数
        """
        # key1とkey2に一致するデータを取得
        data1: np.ndarray = np.array(self.df[key1].values)
        data2: np.ndarray = np.array(self.df[key2].values)

        # 遅れ時間の補正
        if key2 in self.apply_lag_keys:
            data1, data2 = self.__correct_time_lag(self.lag_index, data1, data2)

        data_length: int = len(data1)

        # 共分散の計算
        cov_matrix: np.ndarray = np.cov(data1, data2)
        covariance: float = cov_matrix[0, 1]

        window: np.ndarray = np.array([])  # 空のnumpy配列で初期化
        if self.apply_window:
            # ハニング窓の作成
            window = self.__generate_window_function(
                type=self.window_type, data_length=data_length
            )

            # ハニング窓の適用
            data1 *= window
            data2 *= window

        # 周波数軸の作成
        freqs: np.ndarray = np.fft.rfftfreq(data_length, 1.0 / self.fs)

        # 実数高速フーリエ変換の実行
        fft1: np.ndarray = np.fft.rfft(data1)
        fft2: np.ndarray = np.fft.rfft(data2)

        # コスペクトルの計算
        cospectrum: np.ndarray = 2 * np.real(np.conj(fft1) * fft2) / data_length

        # 窓関数の平均二乗値で割ることでスケーリングを行う
        if self.apply_window:
            cospectrum /= np.mean(window**2)

        # コスペクトルの正規化
        if self.dimensionless:
            cospectrum /= covariance

        # 周波数が0でない要素のみを選択
        nonzero_mask: np.ndarray = freqs != 0
        nonzero_freqs: np.ndarray = freqs[nonzero_mask]
        nonzero_cospectrum: np.ndarray = cospectrum[nonzero_mask]

        # 周波数の加法的重みづけ
        if frequency_weighted:
            nonzero_cospectrum *= nonzero_freqs  # 各周波数に対応する値をかける

        # 周波数軸と正規化コスペクトルの対数を取る
        log_freqs: np.ndarray = np.log10(nonzero_freqs)
        log_cospectrum: np.ndarray = np.log10(np.abs(nonzero_cospectrum))

        # # 周波数の乗法的重みづけ
        # if frequency_weighted:
        #     log_cospectrum *= log_freqs  # 各周波数に対応する値をかける

        if smooth:
            # データ端の処理のため、端点を複製
            padded_data = np.pad(
                log_cospectrum, (window_size // 2, window_size // 2), mode="edge"
            )
            # 移動平均を適用
            log_cospectrum = self.__moving_average(padded_data, window_size)
            # 元の長さにトリミング
            if window_size % 2 == 0:
                log_cospectrum = log_cospectrum[:-1]

        # 周波数軸の最小値と最大値を取得
        min_freq: float = np.min(log_freqs)
        max_freq: float = np.max(log_freqs)

        # 等間隔なplots個の点を生成
        interp_log_freqs: np.ndarray = np.linspace(min_freq, max_freq, self.plots)
        interp_freqs: np.ndarray = 10**interp_log_freqs

        # 生成した周波数に対応するコスペクトルの値を線形補間
        interp_log_cospectrum: np.ndarray = np.interp(
            interp_log_freqs, log_freqs, log_cospectrum
        )
        interp_cospectrum: np.ndarray = 10**interp_log_cospectrum

        # 相関係数の計算
        correlation_coefficient: float = np.corrcoef(data1, data2)[0, 1]

        return interp_freqs, interp_cospectrum, correlation_coefficient
