import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from logging import getLogger, Formatter, Logger, StreamHandler, DEBUG, INFO
from pathlib import Path


@dataclass
class MSAInputConfig:
    """入力ファイルの設定を保持するデータクラス"""

    path: Path | str  # ファイルパス
    delay: int = 0  # 測器の遅れ時間（秒）


@dataclass
class HotspotData:
    """ホットスポットの情報を保持するデータクラス"""

    angle: float  # 中心からの角度
    ratio: float  # ΔC2H6/ΔCH4の比率
    avg_lat: float  # 平均緯度
    avg_lon: float  # 平均経度
    section: int  # 所属する区画番号
    source: str  # データソース
    type: str  # ホットスポットの種類 ("bio" or "city")


class MobileSpatialAnalyzer:
    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        inputs: list[MSAInputConfig] | list[tuple[str | Path, int]],
        num_sections: int,
        sampling_frequency: float = 1.0,  # サンプリング周波数(Hz)
        window_minutes: float = 5.0,  # 移動窓の大きさ（分）
        correlation_threshold: float = 0.7,
        ch4_enhance_threshold: float = 0.1,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        測定データ解析クラスの初期化

        Args:
            center_lat: 中心緯度
            center_lon: 中心経度
            inputs: 入力ファイルのリスト
            num_sections: 分割する区画数
            sampling_frequency: サンプリング周波数(Hz)
            window_minutes: 移動窓の大きさ（分）
            correlation_threshold: 相関係数の閾値
            ch4_enhance_threshold: CH4増加の閾値(ppm)
            logger (Logger | None): 使用するロガー。Noneの場合は新しいロガーを作成します。
            logging_debug (bool): ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。
        """
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.num_sections = num_sections
        self.section_size = 360 / num_sections
        self.correlation_threshold = correlation_threshold
        self.ch4_enhance_threshold = ch4_enhance_threshold
        self.sampling_frequency = sampling_frequency
        self.window_minutes = window_minutes

        # window_sizeをデータポイント数に変換（分→秒→データポイント数）
        self.window_size: int = self.__calculate_window_size(window_minutes)

        # 入力設定の標準化
        self.__input_configs = self.__normalize_inputs(inputs)
        # 複数ファイルのデータを読み込み
        self.__data = self.__load_all_data()
        self.__sections = self.__initialize_sections()

        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = self.__setup_logger(logger, log_level)

    def __setup_logger(self, logger: Logger | None, log_level: int = INFO):
        """
        ロガーを設定します。

        このメソッドは、ロギングの設定を行い、ログメッセージのフォーマットを指定します。
        ログメッセージには、日付、ログレベル、メッセージが含まれます。

        渡されたロガーがNoneまたは不正な場合は、新たにロガーを作成し、標準出力に
        ログメッセージが表示されるようにStreamHandlerを追加します。ロガーのレベルは
        引数で指定されたlog_levelに基づいて設定されます。

        引数:
            logger (Logger | None): 使用するロガー。Noneの場合は新しいロガーを作成します。
            log_level (int): ロガーのログレベル。デフォルトはINFO。

        戻り値:
            Logger: 設定されたロガーオブジェクト。
        """
        if logger is not None and isinstance(logger, Logger):
            return logger
        # 渡されたロガーがNoneまたは正しいものでない場合は独自に設定
        logger: Logger = getLogger()
        logger.setLevel(log_level)  # ロガーのレベルを設定
        ch = StreamHandler()
        ch_formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(ch_formatter)  # フォーマッターをハンドラーに設定
        logger.addHandler(ch)  # StreamHandlerの追加
        return logger

    def __normalize_inputs(
        self, inputs: list[MSAInputConfig] | list[tuple[str | Path, int]]
    ) -> list[MSAInputConfig]:
        """入力設定を標準化"""
        normalized = []
        for inp in inputs:
            if isinstance(inp, MSAInputConfig):
                normalized.append(inp)
            else:
                path, delay = inp
                # 拡張子の確認
                extension = Path(path).suffix
                if extension not in [".txt", ".csv"]:
                    raise ValueError(f"Unsupported file extension: {extension}")
                normalized.append(MSAInputConfig(path=path, delay=delay))
        return normalized

    def __load_all_data(self) -> dict[str, pd.DataFrame]:
        """全入力ファイルのデータを読み込む"""
        all_data = {}
        for config in self.__input_configs:
            df = self.__load_data(config)
            source_name = Path(config.path).stem
            all_data[source_name] = df
        return all_data

    def __load_data(self, config: MSAInputConfig) -> pd.DataFrame:
        """
        測定データの読み込みと前処理

        Args:
            config: 入力ファイルの設定

        Returns:
            pd.DataFrame: 読み込んだデータフレーム
        """
        df = pd.read_csv(config.path, na_values=["No Data", "nan"])

        # カラム名の標準化（測器に依存しない汎用的な名前に変更）
        column_mapping = {
            "Time Stamp": "timestamp",
            "CH4 (ppm)": "ch4_ppm",
            "C2H6 (ppb)": "c2h6_ppb",
            "H2O (ppm)": "h2o_ppm",
            "Latitude": "latitude",
            "Longitude": "longitude",
        }
        df = df.rename(columns=column_mapping)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        if config.delay > 0:
            # 遅れ時間の補正
            columns_to_shift = ["ch4_ppm", "c2h6_ppb", "h2o_ppm"]
            shift_periods = -config.delay

            for col in columns_to_shift:
                df[col] = df[col].shift(shift_periods)

            df = df.dropna(subset=columns_to_shift)

        # 水蒸気フィルタリング
        df[df["h2o_ppm"] < 2000] = np.nan
        df.dropna(subset=["ch4_ppm"], inplace=True)

        return df

    def __initialize_sections(self) -> dict[int, tuple[float, float]]:
        """区画の初期化"""
        sections = {}
        for i in range(self.num_sections):
            # -180から180の範囲で区画を設定
            start_angle = -180 + i * self.section_size
            end_angle = -180 + (i + 1) * self.section_size
            sections[i] = (start_angle, end_angle)
        return sections

    def __calculate_angle(self, lat: float, lon: float) -> float:
        """
        中心からの角度を計算

        Returns:
            float: 真北を0°として時計回りの角度（-180°から180°）
        """
        d_lat = lat - self.center_lat
        d_lon = lon - self.center_lon

        # arctanを使用して角度を計算（ラジアン）
        angle_rad = math.atan2(d_lon, d_lat)

        # ラジアンから度に変換（-180から180の範囲）
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def __calculate_window_size(self, window_minutes: float) -> int:
        """
        時間窓からデータポイント数を計算

        Args:
            window_minutes: 時間窓の大きさ（分）

        Returns:
            int: データポイント数
        """
        return int(60 * window_minutes)

    def __determine_section(self, angle: float) -> int:
        """
        角度から所属する区画を判定

        Returns:
            int: 区画番号
        """
        for section_num, (start, end) in self.__sections.items():
            if start <= angle < end:
                return section_num
        # -180度の場合は最後の区画に含める
        return self.num_sections - 1

    def __detect_hotspots(
        self,
        df: pd.DataFrame,
        ch4_enhance_threshold: float,
        correlation_threshold: float,
        window_size: int,
    ) -> list[HotspotData]:
        """シンプル化したホットスポット検出"""
        hotspots: list[HotspotData] = []

        # CH4増加量が閾値を超えるデータポイントを抽出
        enhanced_mask = df["ch4_ppm"] - df["ch4_ppm_mv"] > ch4_enhance_threshold

        if enhanced_mask.any():
            # 必要なデータを抽出
            lat = df["latitude"][enhanced_mask]
            lon = df["longitude"][enhanced_mask]
            ratios = df["c2h6_ch4_ratio_delta"][enhanced_mask]
            correlations = df["ch4_c2h6_correlation"][enhanced_mask]

            # デバッグ情報の出力
            self.logger.debug(f"{lat};{lon};{ratios}")

            # 各ポイントに対してホットスポットを作成
            for i in range(len(lat)):
                if pd.notna(ratios.iloc[i]):  # 有効な比率データのみ処理
                    angle = self.__calculate_angle(lat.iloc[i], lon.iloc[i])
                    section = self.__determine_section(angle)

                    # 相関係数に基づいてタイプを決定
                    spot_type = (
                        "gas"
                        if correlations.iloc[i] >= correlation_threshold
                        else "bio"
                    )

                    hotspots.append(
                        HotspotData(
                            angle=angle,
                            ratio=ratios.iloc[i],
                            avg_lat=lat.iloc[i],
                            avg_lon=lon.iloc[i],
                            section=section,
                            source=ratios.index[i].strftime("%Y-%m-%d"),
                            type=spot_type,
                        )
                    )

        return hotspots

    def __calculate_hotspots_parameters(
        self, df: pd.DataFrame, window_size: int
    ) -> pd.DataFrame:
        """パラメータ計算"""
        # 移動平均の計算
        df["ch4_ppm_mv"] = (
            df["ch4_ppm"].rolling(window=window_size, center=True, min_periods=1).mean()
        )

        df["c2h6_ppb_mv"] = (
            df["c2h6_ppb"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )

        # 移動相関の計算
        df["ch4_c2h6_correlation"] = (
            df["ch4_ppm"]
            .rolling(window=window_size, min_periods=1)
            .corr(df["c2h6_ppb"])
        )

        # 移動平均からの偏差
        df["ch4_ppm_delta"] = df["ch4_ppm"] - df["ch4_ppm_mv"]
        df["c2h6_ppb_delta"] = df["c2h6_ppb"] - df["c2h6_ppb_mv"]

        # C2H6/CH4の比率計算
        df["c2h6_ch4_ratio"] = df["c2h6_ppb"] / df["ch4_ppm"]

        # デルタ値に基づく比の計算
        ch4_threshold = 0.005  # 閾値
        df["c2h6_ch4_ratio_delta"] = np.where(
            (df["ch4_ppm_delta"].abs() >= ch4_threshold)
            & (df["c2h6_ppb_delta"] >= 0.0),
            df["c2h6_ppb_delta"] / df["ch4_ppm_delta"],
            np.nan,
        )

        return df

    def analyze_hotspots(self) -> list[HotspotData]:
        """
        ホットスポットを検出して分析
        window_sizeはクラス初期化時に設定した値を使用

        Returns:
            list[HotspotData]: 検出されたホットスポットのリスト
        """
        all_hotspots: list[HotspotData] = []

        # 各データソースに対して解析を実行
        for _, df in self.__data.items():
            # パラメータの計算
            df = self.__calculate_hotspots_parameters(df, self.window_size)

            # ホットスポットの検出
            hotspots = self.__detect_hotspots(
                df,
                self.ch4_enhance_threshold,
                self.correlation_threshold,
                self.window_size,
            )
            all_hotspots.extend(hotspots)

        return all_hotspots
