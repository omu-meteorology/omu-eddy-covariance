import math
import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from logging import getLogger, Formatter, Logger, StreamHandler, DEBUG, INFO
from omu_eddy_covariance import HotspotData

"""
堺市役所の位置情報

center_lat=34.573904320329724,
center_lon=135.4829511120712,
"""


@dataclass
class MSAInputConfig:
    """入力ファイルの設定を保持するデータクラス"""

    path: Path | str  # ファイルパス
    delay: float = 0  # 測器の遅れ時間（秒）


class MobileSpatialAnalyzer:
    """
    移動観測で得られた測定データを解析するクラス
    """

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        inputs: list[MSAInputConfig] | list[tuple[str | Path, int]],
        num_sections: int = 4,
        ch4_enhance_threshold: float = 0.1,
        correlation_threshold: float = 0.7,
        hotspot_area_meter: float = 30,
        window_minutes: float = 5,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        測定データ解析クラスの初期化

        Args:
            center_lat (float): 中心緯度
            center_lon (float): 中心経度
            inputs (list[MSAInputConfig] | list[tuple[str | Path, int]]): 入力ファイルのリスト
            num_sections (int): 分割する区画数。デフォルトは4。
            ch4_enhance_threshold (float): CH4増加の閾値(ppm)。デフォルトは0.1。
            correlation_threshold (float): 相関係数の閾値。デフォルトは0.7。
            hotspot_area_meter (float): ホットスポットの検出に使用するエリアの半径（メートル）。デフォルトは30メートル。
            window_minutes (float): 移動窓の大きさ（分）。デフォルトは5分。
            logger (Logger | None): 使用するロガー。Noneの場合は新しいロガーを作成します。
            logging_debug (bool): ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。
        """
        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = MobileSpatialAnalyzer.setup_logger(logger, log_level)
        # プライベートなプロパティ
        self.__center_lat: float = center_lat
        self.__center_lon: float = center_lon
        self.__ch4_enhance_threshold: float = ch4_enhance_threshold
        self.__correlation_threshold: float = correlation_threshold
        self.__hotspot_area_meter: float = hotspot_area_meter
        self.__num_sections: int = num_sections
        # セクションの範囲
        section_size: float = 360 / num_sections
        self.__section_size: float = section_size
        self.__sections = self.__initialize_sections(num_sections, section_size)
        # window_sizeをデータポイント数に変換（分→秒→データポイント数）
        self.__window_size: int = self.__calculate_window_size(window_minutes)
        # 入力設定の標準化
        normalized_input_configs: list[MSAInputConfig] = self.__normalize_inputs(inputs)
        # 複数ファイルのデータを読み込み
        self.__data: dict[str, pd.DataFrame] = self.__load_all_data(
            normalized_input_configs
        )

    @staticmethod
    def setup_logger(logger: Logger | None, log_level: int = INFO):
        """
        ロガーを設定します。

        このメソッドは、ロギングの設定を行い、ログメッセージのフォーマットを指定します。
        ログメッセージには、日付、ログレベル、メッセージが含まれます。

        渡されたロガーがNoneまたは不正な場合は、新たにロガーを作成し、標準出力に
        ログメッセージが表示されるようにStreamHandlerを追加します。ロガーのレベルは
        引数で指定されたlog_levelに基づいて設定されます。

        Args:
            logger (Logger | None): 使用するロガー。Noneの場合は新しいロガーを作成します。
            log_level (int): ロガーのログレベル。デフォルトはINFO。

        Returns:
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

    def analyze_hotspots(
        self,
        exclude_duplicates_across_days: bool = False,
        additional_distance_meter: float = 20,
    ) -> list[HotspotData]:
        """
        ホットスポットを検出して分析します。

        このメソッドは、クラス初期化時に設定されたwindow_sizeを使用して、
        各データソースに対してホットスポットを検出し、分析結果を返します。

        Args:
            exclude_duplicates_across_days (bool): 異なる日付間での重複を除外するかどうか。
                True の場合、全期間で重複するホットスポットを除外します。
                False の場合、日付ごとに独立してホットスポットを検出します。
                デフォルトは False です。
            additional_distance_meter (float): hotspot_area_meterに追加する距離（メートル）。
                重複除外時の距離閾値は hotspot_area_meter + additional_distance_meter となります。
                デフォルトは20メートルです。

        Returns:
            list[HotspotData]: 検出されたホットスポットのリスト。
            各ホットスポットは、位置、比率、タイプなどの情報を含みます。
        """
        all_hotspots: list[HotspotData] = []

        # 各データソースに対して解析を実行
        for _, df in self.__data.items():
            # パラメータの計算
            df = self.__calculate_hotspots_parameters(df, self.__window_size)

            # ホットスポットの検出
            hotspots: list[HotspotData] = self.__detect_hotspots(
                df,
                ch4_enhance_threshold=self.__ch4_enhance_threshold,
                hotspot_areas_meter=self.__hotspot_area_meter,
            )
            all_hotspots.extend(hotspots)

        # 全期間での重複除外が有効な場合
        if exclude_duplicates_across_days:
            distance_threshold: float = (
                self.__hotspot_area_meter + additional_distance_meter
            )
            all_hotspots = self.__remove_duplicates_across_days(
                all_hotspots, distance_threshold_meter=distance_threshold
            )

        return all_hotspots

    def create_hotspots_map(
        self,
        hotspots: list[HotspotData],
        output_dir: str | Path,
        output_filename: str = "hotspots_map",
        center_marker_label: str = "Center",
        plot_center_marker: bool = True,
        radius_meters: float = 3000,
    ) -> None:
        """
        ホットスポットの分布を地図上にプロットして保存

        Args:
            hotspots (list[HotspotData]): プロットするホットスポットのリスト
            output_dir (str | Path): 保存先のディレクトリパス
            output_filename (str): 保存するファイル名。デフォルトは"hotspots_map"。
            center_marker_label (str): 中心を示すマーカーのラベルテキスト。デフォルトは"Center"。
            plot_center_marker (bool): 中心を示すマーカーの有無。デフォルトはTrue。
            radius_meters (float): 区画分けを示す線の長さ。デフォルトは3000。
        """
        output_path: Path = Path(output_dir) / f"{output_filename}.html"
        # 地図の作成
        m = folium.Map(
            location=[self.__center_lat, self.__center_lon],
            zoom_start=15,
            tiles="OpenStreetMap",
        )

        # ホットスポットの種類ごとに異なる色でプロット
        for spot in hotspots:
            # NaN値チェックを追加
            if math.isnan(spot.avg_lat) or math.isnan(spot.avg_lon):
                continue

            # タイプに応じて色を設定
            if spot.type == "comb":
                color = "green"
            elif spot.type == "gas":
                color = "red"
            elif spot.type == "bio":
                color = "blue"
            else:  # invalid type
                color = "black"

            # CSSのgrid layoutを使用してHTMLタグを含むテキストをフォーマット
            popup_html = f"""
            <div style='font-family: Arial; font-size: 12px; display: grid; grid-template-columns: auto auto auto; gap: 5px;'>
                <b>Date</b>    <span>:</span> <span>{spot.source}</span>
                <b>Corr</b>    <span>:</span> <span>{spot.correlation:.3f}</span>
                <b>Ratio</b>   <span>:</span> <span>{spot.ratio:.3f}</span>
                <b>Type</b>    <span>:</span> <span>{spot.type}</span>
                <b>Section</b> <span>:</span> <span>{spot.section}</span>
            </div>
            """

            # ポップアップのサイズを指定
            popup = folium.Popup(
                folium.Html(popup_html, script=True),
                max_width=200,  # 最大幅（ピクセル）
            )

            folium.CircleMarker(
                location=[spot.avg_lat, spot.avg_lon],
                radius=8,
                color=color,
                fill=True,
                popup=popup,
            ).add_to(m)

        # 中心点のマーカー
        if plot_center_marker:
            folium.Marker(
                [self.__center_lat, self.__center_lon],
                popup=center_marker_label,
                icon=folium.Icon(color="green", icon="info-sign"),
            ).add_to(m)

        # 区画の境界線を描画
        for section in range(self.__num_sections):
            start_angle = math.radians(-180 + section * self.__section_size)

            R = 6371000  # 地球の半径（メートル）

            # 境界線の座標を計算
            lat1 = self.__center_lat
            lon1 = self.__center_lon
            lat2 = math.degrees(
                math.asin(
                    math.sin(math.radians(lat1)) * math.cos(radius_meters / R)
                    + math.cos(math.radians(lat1))
                    * math.sin(radius_meters / R)
                    * math.cos(start_angle)
                )
            )
            lon2 = self.__center_lon + math.degrees(
                math.atan2(
                    math.sin(start_angle)
                    * math.sin(radius_meters / R)
                    * math.cos(math.radians(lat1)),
                    math.cos(radius_meters / R)
                    - math.sin(math.radians(lat1)) * math.sin(math.radians(lat2)),
                )
            )

            # 境界線を描画
            folium.PolyLine(
                locations=[[lat1, lon1], [lat2, lon2]],
                color="black",
                weight=1,
                opacity=0.5,
            ).add_to(m)

        # 地図を保存
        m.save(str(output_path))
        self.logger.info(f"地図を保存しました: {output_path}")

    def get_section_size(self) -> float:
        """
        セクションのサイズを取得するメソッド。
        このメソッドは、解析対象のデータを区画に分割する際の
        各区画の角度範囲を示すサイズを返します。

        戻り値:
            float: 1セクションのサイズ（度単位）
        """
        return self.__section_size

    def plot_scatter_c2h6_ch4(
        self,
        output_dir: str | Path,
        output_filename: str = "scatter_c2h6_ch4",
        dpi: int = 200,
        figsize: tuple[int, int] = (4, 4),
        fontsize: float = 12,
        ratio_labels: dict[float, tuple[float, float, str]] | None = None,
        savefig: bool = True,
    ) -> plt.Figure:
        """
        C2H6とCH4の散布図をプロットします。

        Args:
            output_dir (str | Path): 保存先のディレクトリパス
            output_filename (str): 保存するファイル名。デフォルトは"scatter_c2h6_ch4"。
            dpi (int): 解像度。デフォルトは200。
            figsize (tuple[int, int]): 図のサイズ。デフォルトは(4, 4)。
            fontsize (float): フォントサイズ。デフォルトは12。
            ratio_labels (dict[float, tuple[float, float, str]] | None): 比率線とラベルの設定。
                キーは比率値、値は (x位置, y位置, ラベルテキスト) のタプル。
                Noneの場合はデフォルト設定を使用。デフォルト値:
                {
                    0.001: (1.25, 2, "0.001"),
                    0.005: (1.25, 8, "0.005"),
                    0.010: (1.25, 15, "0.01"),
                    0.020: (1.25, 30, "0.02"),
                    0.030: (1.0, 40, "0.03"),
                    0.076: (0.20, 42, "0.076 (Osaka)")
                }
            savefig (bool): 図の保存を許可するフラグ。デフォルトはTrueで、Falseの場合は`output_dir`の指定に関わらず図を保存しない。

        Returns:
            plt.Figure: 作成された散布図のFigureオブジェクト
        """
        output_path: Path = Path(output_dir) / f"{output_filename}.png"

        ch4_enhance_threshold: float = self.__ch4_enhance_threshold
        correlation_threshold: float = self.__correlation_threshold
        data = self.__data

        plt.rcParams["font.size"] = fontsize
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # 全データソースに対してプロット
        for source_name, df in data.items():
            # CH4増加量が閾値未満のデータ
            mask_low = df["ch4_ppm"] - df["ch4_ppm_mv"] < ch4_enhance_threshold
            plt.plot(
                df["ch4_ppm_delta"][mask_low],
                df["c2h6_ppb_delta"][mask_low],
                "o",
                c="gray",
                alpha=0.05,
                ms=2,
                label=f"{source_name} (Low CH4)" if len(data) > 1 else "Low CH4",
            )

            # CH4増加量が閾値以上で、相関が低いデータ
            mask_high_low_corr = (
                df["ch4_c2h6_correlation"] < correlation_threshold
            ) & (df["ch4_ppm"] - df["ch4_ppm_mv"] > ch4_enhance_threshold)
            plt.plot(
                df["ch4_ppm_delta"][mask_high_low_corr],
                df["c2h6_ppb_delta"][mask_high_low_corr],
                "o",
                c="blue",
                alpha=0.5,
                ms=2,
                label=f"{source_name} (Bio)" if len(data) > 1 else "Bio",
            )

            # CH4増加量が閾値以上で、相関が高いデータ
            mask_high_high_corr = (
                df["ch4_c2h6_correlation"] >= correlation_threshold
            ) & (df["ch4_ppm"] - df["ch4_ppm_mv"] > ch4_enhance_threshold)
            plt.plot(
                df["ch4_ppm_delta"][mask_high_high_corr],
                df["c2h6_ppb_delta"][mask_high_high_corr],
                "o",
                c="red",
                alpha=0.5,
                ms=2,
                label=f"{source_name} (Gas)" if len(data) > 1 else "Gas",
            )

        # デフォルトの比率とラベル設定
        default_ratio_labels = {
            0.001: (1.25, 2, "0.001"),
            0.005: (1.25, 8, "0.005"),
            0.010: (1.25, 15, "0.01"),
            0.020: (1.25, 30, "0.02"),
            0.030: (1.0, 40, "0.03"),
            0.076: (0.20, 42, "0.076 (Osaka)"),
        }

        ratio_labels = ratio_labels or default_ratio_labels

        # プロット後、軸の設定前に比率の線を追加
        x = np.array([0, 5])
        base_ch4 = 0.0
        base = 0.0

        # 各比率に対して線を引く
        for ratio, (x_pos, y_pos, label) in ratio_labels.items():
            y = (x - base_ch4) * 1000 * ratio + base
            plt.plot(x, y, "-", c="black", alpha=0.5)
            plt.text(x_pos, y_pos, label)

        # 既存の軸設定を維持
        plt.ylim(0, 50)
        plt.xlim(0, 2.0)
        plt.ylabel("Δ$\\mathregular{C_{2}H_{6}}$ (ppb)")
        plt.xlabel("Δ$\\mathregular{CH_{4}}$ (ppm)")

        # グラフの保存または表示
        if savefig:
            plt.savefig(output_path, bbox_inches="tight")
            self.logger.info(f"散布図を保存しました: {output_path}")

        return fig

    def __calculate_angle(self, lat: float, lon: float) -> float:
        """
        中心からの角度を計算

        Args:
            lat (float): 緯度
            lon (float): 経度

        Returns:
            float: 真北を0°として時計回りの角度（-180°から180°）
        """
        d_lat: float = lat - self.__center_lat
        d_lon: float = lon - self.__center_lon

        # arctanを使用して角度を計算（ラジアン）
        angle_rad: float = math.atan2(d_lon, d_lat)

        # ラジアンから度に変換（-180から180の範囲）
        angle_deg: float = math.degrees(angle_rad)
        return angle_deg

    def __calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        2点間の距離をメートル単位で計算（Haversine formula）

        Args:
            lat1 (float): 地点1の緯度
            lon1 (float): 地点1の経度
            lat2 (float): 地点2の緯度
            lon2 (float): 地点2の経度

        Returns:
            float: 2地点間の距離（メートル）
        """
        R = 6371000  # 地球の半径（メートル）

        # 緯度経度をラジアンに変換
        lat1_rad: float = math.radians(lat1)
        lon1_rad: float = math.radians(lon1)
        lat2_rad: float = math.radians(lat2)
        lon2_rad: float = math.radians(lon2)

        # 緯度と経度の差分
        dlat: float = lat2_rad - lat1_rad
        dlon: float = lon2_rad - lon1_rad

        # Haversine formula
        a: float = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c: float = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c  # メートル単位での距離

    def __calculate_hotspots_parameters(
        self, df: pd.DataFrame, window_size: int
    ) -> pd.DataFrame:
        """パラメータ計算

        Args:
            df (pd.DataFrame): 入力データフレーム
            window_size (int): 移動窓のサイズ

        Returns:
            pd.DataFrame: 計算されたパラメータを含むデータフレーム
        """
        # 各値の閾値
        ch4_threshold: float = 0.05
        c2h6_threshold: float = 0.0

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
        df["c2h6_ch4_ratio_delta"] = np.where(
            (df["ch4_ppm_delta"].abs() >= ch4_threshold)
            & (df["c2h6_ppb_delta"] >= c2h6_threshold),
            df["c2h6_ppb_delta"] / df["ch4_ppm_delta"],
            np.nan,
        )

        return df

    def __calculate_window_size(self, window_minutes: float) -> int:
        """
        時間窓からデータポイント数を計算

        Args:
            window_minutes (float): 時間窓の大きさ（分）

        Returns:
            int: データポイント数
        """
        return int(60 * window_minutes)

    def __correct_h2o_interference_pico(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        水蒸気干渉の補正を行います。
        CH4濃度に対する水蒸気の干渉を補正する2次関数を適用します。

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 水蒸気干渉が補正されたデータフレーム
        """
        # 補正式の係数（実験的に求められた値）
        a: float = 2.0631  # 切片
        b: float = 1.0111e-06  # 1次の係数
        c: float = -1.8683e-10  # 2次の係数

        # 元のデータを保護するためコピーを作成
        df: pd.DataFrame = df.copy()
        # 水蒸気濃度の配列を取得
        h2o: np.ndarray = np.array(df["h2o_ppm"])

        # 補正項の計算
        correction_curve = a + b * h2o + c * h2o * h2o
        max_correction = np.max(correction_curve)
        correction_term = -(correction_curve - max_correction)

        # CH4濃度の補正
        df["ch4_ppm"] = df["ch4_ppm"] + correction_term
        # 極端に低い水蒸気濃度のデータは信頼性が低いため除外
        df.loc[df["h2o_ppm"] < 2000, "ch4_ppm"] = np.nan
        df = df.dropna(subset=["ch4_ppm"])

        return df

    def __detect_hotspots(
        self,
        df: pd.DataFrame,
        ch4_enhance_threshold: float,
        hotspot_areas_meter: float,
    ) -> list[HotspotData]:
        """シンプル化したホットスポット検出

        Args:
            df (pd.DataFrame): 入力データフレーム
            ch4_enhance_threshold (float): CH4増加の閾値
            hotspot_areas_meter (float): ホットスポット間の最小距離（メートル）

        Returns:
            list[HotspotData]: 検出されたホットスポットのリスト
        """
        hotspots: list[HotspotData] = []
        # タイプごとに使用された位置を記録
        used_positions_by_type: dict[str, set] = {
            "bio": set(),
            "gas": set(),
            "comb": set(),
        }

        # CH4増加量が閾値を超えるデータポイントを抽出
        enhanced_mask = df["ch4_ppm"] - df["ch4_ppm_mv"] > ch4_enhance_threshold

        if enhanced_mask.any():
            # 必要なデータを抽出
            lat = df["latitude"][enhanced_mask]
            lon = df["longitude"][enhanced_mask]
            ratios = df["c2h6_ch4_ratio_delta"][enhanced_mask]

            # デバッグ情報の出力
            self.logger.debug(f"{lat};{lon};{ratios}")

            # 各ポイントに対してホットスポットを作成
            for i in range(len(lat)):
                if pd.notna(ratios.iloc[i]):
                    current_lat = lat.iloc[i]
                    current_lon = lon.iloc[i]
                    correlation = df["ch4_c2h6_correlation"].iloc[i]

                    # 比率に基づいてタイプを決定
                    spot_type = "bio"
                    if ratios.iloc[i] >= 100:
                        spot_type = "comb"
                    elif ratios.iloc[i] >= 5:
                        spot_type = "gas"

                    # 同じタイプのホットスポットとの距離のみをチェック
                    too_close: bool = False
                    for used_lat, used_lon in used_positions_by_type[spot_type]:
                        distance = self.__calculate_distance(
                            current_lat, current_lon, used_lat, used_lon
                        )
                        if distance < hotspot_areas_meter:
                            too_close = True
                            break

                    if too_close:
                        continue

                    angle: float = self.__calculate_angle(current_lat, current_lon)
                    section: int = self.__determine_section(angle)

                    hotspots.append(
                        HotspotData(
                            angle=angle,
                            avg_lat=current_lat,
                            avg_lon=current_lon,
                            correlation=correlation,
                            ratio=ratios.iloc[i],
                            section=section,
                            source=ratios.index[i].strftime("%Y-%m-%d %H:%M:%S"),
                            type=spot_type,
                        )
                    )

                    # タイプごとに使用した位置を記録
                    used_positions_by_type[spot_type].add((current_lat, current_lon))

        return hotspots

    def __determine_section(self, angle: float) -> int:
        """
        角度から所属する区画を判定

        Args:
            angle (float): 計算された角度

        Returns:
            int: 区画番号
        """
        for section_num, (start, end) in self.__sections.items():
            if start <= angle < end:
                return section_num
        # -180度の場合は最後の区画に含める
        return self.__num_sections - 1

    def __initialize_sections(
        self, num_sections: int, section_size: float
    ) -> dict[int, tuple[float, float]]:
        """指定された区画数と区画サイズに基づいて、区画の範囲を初期化します。

        Args:
            num_sections (int): 初期化する区画の数。
            section_size (float): 各区画の角度範囲のサイズ。

        Returns:
            dict[int, tuple[float, float]]: 区画番号とその範囲の辞書。各区画は-180度から180度の範囲に分割されます。
        """
        sections: dict[int, tuple[float, float]] = {}
        for i in range(num_sections):
            # -180から180の範囲で区画を設定
            start_angle = -180 + i * section_size
            end_angle = -180 + (i + 1) * section_size
            sections[i] = (start_angle, end_angle)
        return sections

    def __load_all_data(
        self, input_configs: list[MSAInputConfig]
    ) -> dict[str, pd.DataFrame]:
        """全入力ファイルのデータを読み込み、データフレームの辞書を返します。

        このメソッドは、指定された入力設定に基づいてすべてのデータファイルを読み込み、
        各ファイルのデータをデータフレームとして格納した辞書を生成します。

        Args:
            input_configs (list[MSAInputConfig]): 読み込むファイルの設定リスト。

        Returns:
            dict[str, pd.DataFrame]: 読み込まれたデータフレームの辞書。キーはファイル名、値はデータフレーム。
        """
        all_data: dict[str, pd.DataFrame] = {}
        for config in input_configs:
            df: pd.DataFrame = self.__load_data(config)
            source_name: str = Path(config.path).stem
            all_data[source_name] = df
        return all_data

    def __load_data(self, config: MSAInputConfig) -> pd.DataFrame:
        """
        測定データの読み込みと前処理

        Args:
            config (MSAInputConfig): 入力ファイルの設定

        Returns:
            pd.DataFrame: 読み込んだデータフレーム
        """
        df: pd.DataFrame = pd.read_csv(config.path, na_values=["No Data", "nan"])

        # カラム名の標準化（測器に依存しない汎用的な名前に変更）
        column_mapping: dict[str, str] = {
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

        # 緯度経度のnanを削除
        df = df.dropna(subset=["latitude", "longitude"])

        if config.delay > 0:
            # 遅れ時間の補正
            columns_to_shift: list[str] = ["ch4_ppm", "c2h6_ppb", "h2o_ppm"]
            shift_periods: float = -config.delay

            for col in columns_to_shift:
                df[col] = df[col].shift(shift_periods)

            df = df.dropna(subset=columns_to_shift)

        # 水蒸気干渉の補正を適用
        df = self.__correct_h2o_interference_pico(df)

        return df

    def __normalize_inputs(
        self, inputs: list[MSAInputConfig] | list[tuple[str | Path, int]]
    ) -> list[MSAInputConfig]:
        """入力設定を標準化

        Args:
            inputs (list[MSAInputConfig] | list[tuple[str | Path, int]]): 入力設定のリスト

        Returns:
            list[MSAInputConfig]: 標準化された入力設定のリスト
        """
        normalized: list[MSAInputConfig] = []
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

    def __remove_duplicates_across_days(
        self,
        hotspots: list[HotspotData],
        distance_threshold_meter: float,
    ) -> list[HotspotData]:
        """
        全期間での重複するホットスポットを除外します。

        Args:
            hotspots (list[HotspotData]): 元のホットスポットのリスト
            distance_threshold_meter (float): 重複とみなす距離の閾値（メートル）

        Returns:
            list[HotspotData]: 重複を除外したホットスポットのリスト
        """
        # 日付でソート（古い順）
        sorted_hotspots: list[HotspotData] = sorted(hotspots, key=lambda x: x.source)

        # タイプごとに使用された位置を記録
        used_positions_by_type: dict[str, set] = {
            "bio": set(),
            "gas": set(),
            "comb": set(),
        }
        unique_hotspots: list[HotspotData] = []

        for spot in sorted_hotspots:
            # 同じタイプのホットスポットとの距離をチェック
            too_close: bool = False
            for used_lat, used_lon in used_positions_by_type[spot.type]:
                distance: float = self.__calculate_distance(
                    spot.avg_lat, spot.avg_lon, used_lat, used_lon
                )
                if distance < distance_threshold_meter:
                    too_close = True
                    self.logger.debug(
                        f"重複を検出: {spot.source} ({spot.avg_lat}, {spot.avg_lon})"
                        f" - タイプ: {spot.type}, 距離: {distance:.1f}m"
                    )
                    break

            if not too_close:
                unique_hotspots.append(spot)
                used_positions_by_type[spot.type].add((spot.avg_lat, spot.avg_lon))

        self.logger.info(
            f"重複除外: {len(hotspots)} → {len(unique_hotspots)} ホットスポット"
        )
        return unique_hotspots
