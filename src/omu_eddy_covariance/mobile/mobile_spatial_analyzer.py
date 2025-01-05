import os
import math
import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta
from dataclasses import dataclass
from logging import getLogger, Formatter, Logger, StreamHandler, DEBUG, INFO
from ..commons.hotspot_data import HotspotData

"""
堺市役所の位置情報

center_lat=34.573904320329724,
center_lon=135.4829511120712,
"""


@dataclass
class MSAInputConfig:
    """入力ファイルの設定を保持するデータクラス"""

    fs: float  # サンプリング周波数（Hz）
    lag: float  # 測器の遅れ時間（秒）
    path: Path | str  # ファイルパス

    def __post_init__(self) -> None:
        """
        インスタンス生成後に入力値の検証を行います。

        Raises:
            ValueError: 遅延時間が負の値である場合、またはサポートされていないファイル拡張子の場合。
        """
        # fsが有効かを確認
        if not isinstance(self.fs, (int, float)) or self.fs <= 0:
            raise ValueError(
                f"Invalid sampling frequency: {self.fs}. Must be a positive float."
            )
        # lagが0以上のfloatかを確認
        if not isinstance(self.lag, (int, float)) or self.lag < 0:
            raise ValueError(
                f"Invalid lag value: {self.lag}. Must be a non-negative float."
            )
        # 拡張子の確認
        supported_extensions: list[str] = [".txt", ".csv"]
        extension = Path(self.path).suffix
        if extension not in supported_extensions:
            raise ValueError(
                f"Unsupported file extension: '{extension}'. Supported: {supported_extensions}"
            )

    @classmethod
    def validate_and_create(
        cls,
        fs: float,
        lag: float,
        path: Path | str,
    ) -> "MSAInputConfig":
        """
        入力値を検証し、MSAInputConfigインスタンスを生成するファクトリメソッド。

        このメソッドは、指定された遅延時間、サンプリング周波数、およびファイルパスが有効であることを確認し、
        有効な場合に新しいMSAInputConfigオブジェクトを返します。

        Args:
            fs (float): サンプリング周波数。正のfloatである必要があります。
            lag (float): 遅延時間。0以上のfloatである必要があります。
            path (Path | str): 入力ファイルのパス。サポートされている拡張子は.txtと.csvです。

        Returns:
            MSAInputConfig: 検証された入力設定を持つMSAInputConfigオブジェクト。
        """
        return cls(fs=fs, lag=lag, path=path)


class MobileSpatialAnalyzer:
    """
    移動観測で得られた測定データを解析するクラス
    """

    EARTH_RADIUS_METERS: float = 6371000  # 地球の半径（メートル）

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        inputs: list[MSAInputConfig] | list[tuple[float, float, str | Path]],
        num_sections: int = 4,
        ch4_enhance_threshold: float = 0.1,
        correlation_threshold: float = 0.7,
        hotspot_area_meter: float = 50,
        window_minutes: float = 5,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        測定データ解析クラスの初期化

        Args:
            center_lat (float): 中心緯度
            center_lon (float): 中心経度
            inputs (list[MSAInputConfig] | list[tuple[float, float, str | Path]]): 入力ファイルのリスト
            num_sections (int): 分割する区画数。デフォルトは4。
            ch4_enhance_threshold (float): CH4増加の閾値(ppm)。デフォルトは0.1。
            correlation_threshold (float): 相関係数の閾値。デフォルトは0.7。
            hotspot_area_meter (float): ホットスポットの検出に使用するエリアの半径（メートル）。デフォルトは50メートル。
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
        self._center_lat: float = center_lat
        self._center_lon: float = center_lon
        self._ch4_enhance_threshold: float = ch4_enhance_threshold
        self._correlation_threshold: float = correlation_threshold
        self._hotspot_area_meter: float = hotspot_area_meter
        self._num_sections: int = num_sections
        # セクションの範囲
        section_size: float = 360 / num_sections
        self._section_size: float = section_size
        self._sections = MobileSpatialAnalyzer._initialize_sections(
            num_sections, section_size
        )
        # window_sizeをデータポイント数に変換（分→秒→データポイント数）
        self._window_size: int = MobileSpatialAnalyzer._calculate_window_size(
            window_minutes
        )
        # 入力設定の標準化
        normalized_input_configs: list[MSAInputConfig] = (
            MobileSpatialAnalyzer._normalize_inputs(inputs)
        )
        # 複数ファイルのデータを読み込み
        self._data: dict[str, pd.DataFrame] = self._load_all_data(
            normalized_input_configs
        )

    def analyze_delta_ch4_stats(self, hotspots: list[HotspotData]) -> None:
        """
        各タイプのホットスポットについてΔCH4の統計情報を計算し、結果を表示します。

        Args:
            hotspots (list[HotspotData]): 分析対象のホットスポットリスト
        """
        # タイプごとにホットスポットを分類
        hotspots_by_type = {
            "bio": [h for h in hotspots if h.type == "bio"],
            "gas": [h for h in hotspots if h.type == "gas"],
            "comb": [h for h in hotspots if h.type == "comb"],
        }

        # 統計情報を計算し、表示
        for spot_type, spots in hotspots_by_type.items():
            if spots:
                delta_ch4_values = [spot.delta_ch4 for spot in spots]
                max_value = max(delta_ch4_values)
                mean_value = sum(delta_ch4_values) / len(delta_ch4_values)
                median_value = sorted(delta_ch4_values)[len(delta_ch4_values) // 2]
                print(f"{spot_type}タイプのホットスポットの統計情報:")
                print(f"  最大値: {max_value}")
                print(f"  平均値: {mean_value}")
                print(f"  中央値: {median_value}")
            else:
                print(f"{spot_type}タイプのホットスポットは存在しません。")

    def analyze_hotspots(
        self,
        duplicate_check_mode: str = "none",
        min_time_threshold_seconds: float = 300,
        max_time_threshold_hours: float = 12,
    ) -> list[HotspotData]:
        """
        ホットスポットを検出して分析します。

        Args:
            duplicate_check_mode (str): 重複チェックのモード。
                - "none": 重複チェックを行わない。
                - "time_window": 指定された時間窓内の重複のみを除外。
                - "time_all": すべての時間範囲で重複チェックを行う。
            min_time_threshold_seconds (float): 重複とみなす最小時間の閾値（秒）。デフォルトは300秒。
            max_time_threshold_hours (float): 重複チェックを一時的に無視する最大時間の閾値（時間）。デフォルトは12時間。

        Returns:
            list[HotspotData]: 検出されたホットスポットのリスト。
        """
        # 不正な入力値に対するエラーチェック
        valid_modes = {"none", "time_window", "time_all"}
        if duplicate_check_mode not in valid_modes:
            raise ValueError(
                f"無効な重複チェックモード: {duplicate_check_mode}. 有効な値は {valid_modes} です。"
            )

        all_hotspots: list[HotspotData] = []

        # 各データソースに対して解析を実行
        for _, df in self._data.items():
            # パラメータの計算
            df = self._calculate_hotspots_parameters(df, self._window_size)

            # ホットスポットの検出
            hotspots: list[HotspotData] = self._detect_hotspots(
                df,
                ch4_enhance_threshold=self._ch4_enhance_threshold,
            )
            all_hotspots.extend(hotspots)

        # 重複チェックモードに応じて処理
        if duplicate_check_mode != "none":
            all_hotspots = self._remove_duplicates(
                all_hotspots,
                check_time_all=duplicate_check_mode == "time_all",
                min_time_threshold_seconds=min_time_threshold_seconds,
                max_time_threshold_hours=max_time_threshold_hours,
            )

        return all_hotspots

    def calculate_measurement_stats(
        self,
        show_individual_stats: bool = True,
        show_total_stats: bool = True,
    ) -> tuple[float, timedelta]:
        """
        各ファイルの測定時間と走行距離を計算し、合計を返します。

        Args:
            show_individual_stats (bool): 個別ファイルの統計を表示するかどうか。デフォルトはTrue。
            show_total_stats (bool): 合計統計を表示するかどうか。デフォルトはTrue。

        Returns:
            tuple[float, timedelta]: 総距離(km)と総時間のタプル
        """
        total_distance: float = 0.0
        total_time: timedelta = timedelta()
        individual_stats: list[dict] = []  # 個別の統計情報を保存するリスト

        # プログレスバーを表示しながら計算
        for source_name, df in tqdm(
            self._data.items(), desc="Calculating", unit="file"
        ):
            # 時間の計算
            time_spent = df.index[-1] - df.index[0]

            # 距離の計算
            distance_km = 0.0
            for i in range(len(df) - 1):
                lat1, lon1 = df.iloc[i][["latitude", "longitude"]]
                lat2, lon2 = df.iloc[i + 1][["latitude", "longitude"]]
                distance_km += (
                    MobileSpatialAnalyzer._calculate_distance(
                        lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2
                    )
                    / 1000
                )

            # 合計に加算
            total_distance += distance_km
            total_time += time_spent

            # 統計情報を保存
            if show_individual_stats:
                average_speed = distance_km / (time_spent.total_seconds() / 3600)
                individual_stats.append(
                    {
                        "source": source_name,
                        "distance": distance_km,
                        "time": time_spent,
                        "speed": average_speed,
                    }
                )

        # 計算完了後に統計情報を表示
        if show_individual_stats:
            self.logger.info("=== Individual Stats ===")
            for stat in individual_stats:
                print(f"File         : {stat['source']}")
                print(f"  Distance   : {stat['distance']:.2f} km")
                print(f"  Time       : {stat['time']}")
                print(f"  Avg. Speed : {stat['speed']:.1f} km/h\n")

        # 合計を表示
        if show_total_stats:
            average_speed_total: float = total_distance / (
                total_time.total_seconds() / 3600
            )
            self.logger.info("=== Total Stats ===")
            print(f"  Distance   : {total_distance:.2f} km")
            print(f"  Time       : {total_time}")
            print(f"  Avg. Speed : {average_speed_total:.1f} km/h\n")

        return total_distance, total_time

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
            location=[self._center_lat, self._center_lon],
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
                [self._center_lat, self._center_lon],
                popup=center_marker_label,
                icon=folium.Icon(color="green", icon="info-sign"),
            ).add_to(m)

        # 区画の境界線を描画
        for section in range(self._num_sections):
            start_angle = math.radians(-180 + section * self._section_size)

            R = self.EARTH_RADIUS_METERS

            # 境界線の座標を計算
            lat1 = self._center_lat
            lon1 = self._center_lon
            lat2 = math.degrees(
                math.asin(
                    math.sin(math.radians(lat1)) * math.cos(radius_meters / R)
                    + math.cos(math.radians(lat1))
                    * math.sin(radius_meters / R)
                    * math.cos(start_angle)
                )
            )
            lon2 = self._center_lon + math.degrees(
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

    def export_hotspots_to_csv(
        self,
        hotspots: list[HotspotData],
        output_dir: str | Path,
        output_filename: str = "hotspots.csv",
    ) -> None:
        """
        ホットスポットの情報をCSVファイルに出力します。

        Args:
            hotspots (list[HotspotData]): 出力するホットスポットのリスト
            output_dir (str | Path): 出力先ディレクトリ
            filename (str): 出力ファイル名
        """
        # 日時の昇順でソート
        sorted_hotspots = sorted(hotspots, key=lambda x: x.source)

        # 出力用のデータを作成
        records = []
        for spot in sorted_hotspots:
            record = {
                "source": spot.source,
                "type": spot.type,
                "delta_ch4": spot.delta_ch4,
                "delta_c2h6": spot.delta_c2h6,
                "ratio": spot.ratio,
                "correlation": spot.correlation,
                "angle": spot.angle,
                "section": spot.section,
                "latitude": spot.avg_lat,
                "longitude": spot.avg_lon,
            }
            records.append(record)

        # DataFrameに変換してCSVに出力
        output_path: str = os.path.join(output_dir, output_filename)
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        self.logger.info(
            f"ホットスポット情報をCSVファイルに出力しました: {output_path}"
        )

    def get_section_size(self) -> float:
        """
        セクションのサイズを取得するメソッド。
        このメソッドは、解析対象のデータを区画に分割する際の
        各区画の角度範囲を示すサイズを返します。

        Returns:
            float: 1セクションのサイズ（度単位）
        """
        return self._section_size

    def plot_ch4_delta_histogram(
        self,
        hotspots: list[HotspotData],
        output_dir: str | Path,
        output_filename: str = "ch4_delta_histogram",
        dpi: int = 200,
        figsize: tuple[int, int] = (8, 6),
        fontsize: float = 20,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        save_fig: bool = True,
        show_fig: bool = True,
        yscale_log: bool = True,
    ) -> None:
        """
        CH4の増加量（ΔCH4）の積み上げヒストグラムをプロットします。

        Args:
            hotspots (list[HotspotData]): プロットするホットスポットのリスト
            output_dir (str | Path): 保存先のディレクトリパス
            output_filename (str): 保存するファイル名。デフォルトは"ch4_delta_histogram"。
            dpi (int): 解像度。デフォルトは200。
            figsize (tuple[int, int]): 図のサイズ。デフォルトは(8, 6)。
            fontsize (float): フォントサイズ。デフォルトは20。
            xlim (tuple[float, float] | None): x軸の範囲。Noneの場合は自動設定。
            ylim (tuple[float, float] | None): y軸の範囲。Noneの場合は自動設定。
            save_fig (bool): 図の保存を許可するフラグ。デフォルトはTrue。
            show_fig (bool): 図の表示を許可するフラグ。デフォルトはTrue。
            yscale_log (bool): y軸をlogにするかどうか。デフォルトはTrue。
        """
        output_path: Path = Path(output_dir) / f"{output_filename}.png"

        plt.rcParams["font.size"] = fontsize
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # ホットスポットからデータを抽出
        all_ch4_deltas = []
        all_types = []
        for spot in hotspots:
            all_ch4_deltas.append(spot.delta_ch4)
            all_types.append(spot.type)

        # データをNumPy配列に変換
        all_ch4_deltas = np.array(all_ch4_deltas)
        all_types = np.array(all_types)

        # 0.1刻みのビンを作成
        if xlim is not None:
            bins = np.arange(xlim[0], xlim[1] + 0.1, 0.1)
        else:
            max_val = np.ceil(np.max(all_ch4_deltas) * 10) / 10
            bins = np.arange(0, max_val + 0.1, 0.1)

        # タイプごとのヒストグラムデータを計算
        hist_data = {}
        for type_name in ["bio", "gas", "comb"]:
            mask = all_types == type_name
            if np.any(mask):
                counts, _ = np.histogram(all_ch4_deltas[mask], bins=bins)
                hist_data[type_name] = counts

        # 積み上げヒストグラムを作成
        colors = {"bio": "blue", "gas": "red", "comb": "green"}
        bottom = np.zeros_like(hist_data.get("bio", np.zeros(len(bins) - 1)))

        for type_name in ["bio", "gas", "comb"]:
            if type_name in hist_data:
                plt.bar(
                    bins[:-1],
                    hist_data[type_name],
                    width=np.diff(bins)[0],
                    bottom=bottom,
                    color=colors[type_name],
                    label=type_name,
                    alpha=0.6,
                    align="edge",
                )
                bottom += hist_data[type_name]

        if yscale_log:
            plt.yscale("log")
        plt.xlabel("Δ$\\mathregular{CH_{4}}$ (ppm)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)

        # 軸の範囲を設定
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        # グラフの保存または表示
        if save_fig:
            plt.savefig(output_path, bbox_inches="tight")
            self.logger.info(f"ヒストグラムを保存しました: {output_path}")

        if show_fig:
            plt.show()

        plt.close(fig)

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

        ch4_enhance_threshold: float = self._ch4_enhance_threshold
        correlation_threshold: float = self._correlation_threshold
        data = self._data

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

    def _calculate_hotspots_parameters(
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

    def _correct_h2o_interference_pico(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        水蒸気干渉の補正を行います。
        CH4濃度に対する水蒸気の干渉を補正する2次関数を適用します。

        参考文献:
            Commane et al. (2023): Intercomparison of commercial analyzers for atmospheric ethane and methane observations
                https://amt.copernicus.org/articles/16/1431/2023/,
                https://amt.copernicus.org/articles/16/1431/2023/amt-16-1431-2023.pdf

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

    def _detect_hotspots(
        self,
        df: pd.DataFrame,
        ch4_enhance_threshold: float,
    ) -> list[HotspotData]:
        """シンプル化したホットスポット検出

        Args:
            df (pd.DataFrame): 入力データフレーム
            ch4_enhance_threshold (float): CH4増加の閾値

        Returns:
            list[HotspotData]: 検出されたホットスポットのリスト
        """
        hotspots: list[HotspotData] = []

        # CH4増加量が閾値を超えるデータポイントを抽出
        enhanced_mask = df["ch4_ppm_delta"] >= ch4_enhance_threshold

        if enhanced_mask.any():
            lat = df["latitude"][enhanced_mask]
            lon = df["longitude"][enhanced_mask]
            ratios = df["c2h6_ch4_ratio_delta"][enhanced_mask]
            delta_ch4 = df["ch4_ppm_delta"][enhanced_mask]
            delta_c2h6 = df["c2h6_ppb_delta"][enhanced_mask]

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

                    angle: float = MobileSpatialAnalyzer._calculate_angle(
                        lat=current_lat,
                        lon=current_lon,
                        center_lat=self._center_lat,
                        center_lon=self._center_lon,
                    )
                    section: int = self._determine_section(angle)

                    hotspots.append(
                        HotspotData(
                            source=ratios.index[i].strftime("%Y-%m-%d %H:%M:%S"),
                            angle=angle,
                            avg_lat=current_lat,
                            avg_lon=current_lon,
                            delta_ch4=delta_ch4.iloc[i],
                            delta_c2h6=delta_c2h6.iloc[i],
                            correlation=max(-1, min(1, correlation)),
                            ratio=ratios.iloc[i],
                            section=section,
                            type=spot_type,
                        )
                    )

        return hotspots

    def _determine_section(self, angle: float) -> int:
        """
        角度から所属する区画を判定

        Args:
            angle (float): 計算された角度

        Returns:
            int: 区画番号（0-based-index）
        """
        for section_num, (start, end) in self._sections.items():
            if start <= angle < end:
                return section_num
        # -180度の場合は最後の区画に含める
        return self._num_sections - 1

    def _load_all_data(
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
            df: pd.DataFrame = self._load_data(config)
            source_name: str = Path(config.path).stem
            all_data[source_name] = df
        return all_data

    def _load_data(self, config: MSAInputConfig) -> pd.DataFrame:
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

        if config.lag < 0:
            raise ValueError(
                f"Invalid lag value: {config.lag}. Must be a non-negative float."
            )

        # 遅れ時間の補正
        columns_to_shift: list[str] = ["ch4_ppm", "c2h6_ppb", "h2o_ppm"]
        shift_periods: float = -config.lag

        for col in columns_to_shift:
            df[col] = df[col].shift(shift_periods)

        df = df.dropna(subset=columns_to_shift)

        # 水蒸気干渉の補正を適用
        df = self._correct_h2o_interference_pico(df)

        return df

    def _remove_duplicates(
        self,
        hotspots: list[HotspotData],
        check_time_all: bool,
        min_time_threshold_seconds: float,
        max_time_threshold_hours: float,
    ) -> list[HotspotData]:
        """
        重複するホットスポットを除外します。ΔCH4が大きい順に処理し、
        重複範囲内で最も大きい値を持つホットスポットを残します。

        時間による重複判定の仕様:
        1. min_time_threshold_seconds以内の重複は常に除外
        2. min_time_threshold_seconds < 時間差 <= max_time_threshold_hoursの場合は保持
        3. max_time_threshold_hours以上離れた場合はcheck_time_allパラメータに従う

        Args:
            hotspots (list[HotspotData]): 元のホットスポットのリスト
            check_time_all (bool): max_time_threshold_hours以上離れた場合も重複チェックを継続するかどうか
            min_time_threshold_seconds (float): 重複とみなす最小時間の閾値（秒）
            max_time_threshold_hours (float): 重複チェックを一時的に無視する最大時間の閾値（時間）

        Returns:
            list[HotspotData]: 重複を除外したホットスポットのリスト
        """
        # ΔCH4の降順でソート
        sorted_hotspots: list[HotspotData] = sorted(
            hotspots, key=lambda x: x.delta_ch4, reverse=True
        )
        used_positions_by_type: dict[str, list[tuple[float, float, str, float]]] = {
            "bio": [],
            "gas": [],
            "comb": [],
        }
        unique_hotspots: list[HotspotData] = []

        for spot in sorted_hotspots:
            should_add: bool = True
            for used_lat, used_lon, used_time, used_delta_ch4 in used_positions_by_type[
                spot.type
            ]:
                # 距離チェック
                distance: float = MobileSpatialAnalyzer._calculate_distance(
                    lat1=spot.avg_lat, lon1=spot.avg_lon, lat2=used_lat, lon2=used_lon
                )

                if distance < self._hotspot_area_meter:
                    # 時間差の計算（秒単位）
                    time_diff = pd.Timedelta(
                        pd.to_datetime(spot.source) - pd.to_datetime(used_time)
                    ).total_seconds()
                    time_diff_abs = abs(time_diff)

                    # 時間差に基づく判定
                    if time_diff_abs <= min_time_threshold_seconds:
                        # Case 1: 最小時間閾値以内は常に重複とみなす
                        # ΔCH4が大きい方を残す（現在のスポットは必ず小さい）
                        should_add = False
                        break
                    elif time_diff_abs <= max_time_threshold_hours * 3600:
                        # Case 2: 最大時間閾値以内は重複とみなさない（異なるポイントとして保持）
                        continue
                    elif check_time_all:
                        # Case 3: 最大時間閾値を超えた場合はcheck_time_allに従う
                        # ΔCH4が大きい方を残す（現在のスポットは必ず小さい）
                        should_add = False
                        break

            if should_add:
                unique_hotspots.append(spot)
                used_positions_by_type[spot.type].append(
                    (spot.avg_lat, spot.avg_lon, spot.source, spot.delta_ch4)
                )

        self.logger.info(
            f"重複除外: {len(hotspots)} → {len(unique_hotspots)} ホットスポット"
        )
        return unique_hotspots

    @staticmethod
    def _calculate_angle(
        lat: float, lon: float, center_lat: float, center_lon: float
    ) -> float:
        """
        中心からの角度を計算

        Args:
            lat (float): 対象地点の緯度
            lon (float): 対象地点の経度
            center_lat (float): 中心の緯度
            center_lon (float): 中心の経度

        Returns:
            float: 真北を0°として時計回りの角度（-180°から180°）
        """
        d_lat: float = lat - center_lat
        d_lon: float = lon - center_lon
        # arctanを使用して角度を計算（ラジアン）
        angle_rad: float = math.atan2(d_lon, d_lat)
        # ラジアンから度に変換（-180から180の範囲）
        angle_deg: float = math.degrees(angle_rad)
        return angle_deg

    @classmethod
    def _calculate_distance(
        cls, lat1: float, lon1: float, lat2: float, lon2: float
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
        R = cls.EARTH_RADIUS_METERS

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

    @staticmethod
    def _calculate_window_size(window_minutes: float) -> int:
        """
        時間窓からデータポイント数を計算

        Args:
            window_minutes (float): 時間窓の大きさ（分）

        Returns:
            int: データポイント数
        """
        return int(60 * window_minutes)

    @staticmethod
    def _initialize_sections(
        num_sections: int, section_size: float
    ) -> dict[int, tuple[float, float]]:
        """指定された区画数と区画サイズに基づいて、区画の範囲を初期化します。

        Args:
            num_sections (int): 初期化する区画の数。
            section_size (float): 各区画の角度範囲のサイズ。

        Returns:
            dict[int, tuple[float, float]]: 区画番号（0-based-index）とその範囲の辞書。各区画は-180度から180度の範囲に分割されます。
        """
        sections: dict[int, tuple[float, float]] = {}
        for i in range(num_sections):
            # -180から180の範囲で区画を設定
            start_angle = -180 + i * section_size
            end_angle = -180 + (i + 1) * section_size
            sections[i] = (start_angle, end_angle)
        return sections

    @staticmethod
    def _normalize_inputs(
        inputs: list[MSAInputConfig] | list[tuple[float, float, str | Path]],
    ) -> list[MSAInputConfig]:
        """入力設定を標準化

        Args:
            inputs (list[MSAInputConfig] | list[tuple[float, float, str | Path]]): 入力設定のリスト

        Returns:
            list[MSAInputConfig]: 標準化された入力設定のリスト
        """
        normalized: list[MSAInputConfig] = []
        for inp in inputs:
            if isinstance(inp, MSAInputConfig):
                normalized.append(inp)  # すでに検証済みのため、そのまま追加
            else:
                fs, lag, path = inp
                normalized.append(
                    MSAInputConfig.validate_and_create(fs=fs, lag=lag, path=path)
                )
        return normalized

    @staticmethod
    def _calculate_angle(
        lat: float, lon: float, center_lat: float, center_lon: float
    ) -> float:
        """
        中心からの角度を計算

        Args:
            lat (float): 対象地点の緯度
            lon (float): 対象地点の経度
            center_lat (float): 中心の緯度
            center_lon (float): 中心の経度

        Returns:
            float: 真北を0°として時計回りの角度（-180°から180°）
        """
        d_lat: float = lat - center_lat
        d_lon: float = lon - center_lon
        # arctanを使用して角度を計算（ラジアン）
        angle_rad: float = math.atan2(d_lon, d_lat)
        # ラジアンから度に変換（-180から180の範囲）
        angle_deg: float = math.degrees(angle_rad)
        return angle_deg

    @classmethod
    def _calculate_distance(
        cls, lat1: float, lon1: float, lat2: float, lon2: float
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
        R = cls.EARTH_RADIUS_METERS

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

    @staticmethod
    def _calculate_window_size(window_minutes: float) -> int:
        """
        時間窓からデータポイント数を計算

        Args:
            window_minutes (float): 時間窓の大きさ（分）

        Returns:
            int: データポイント数
        """
        return int(60 * window_minutes)

    @staticmethod
    def _initialize_sections(
        num_sections: int, section_size: float
    ) -> dict[int, tuple[float, float]]:
        """指定された区画数と区画サイズに基づいて、区画の範囲を初期化します。

        Args:
            num_sections (int): 初期化する区画の数。
            section_size (float): 各区画の角度範囲のサイズ。

        Returns:
            dict[int, tuple[float, float]]: 区画番号（0-based-index）とその範囲の辞書。各区画は-180度から180度の範囲に分割されます。
        """
        sections: dict[int, tuple[float, float]] = {}
        for i in range(num_sections):
            # -180から180の範囲で区画を設定
            start_angle = -180 + i * section_size
            end_angle = -180 + (i + 1) * section_size
            sections[i] = (start_angle, end_angle)
        return sections

    @staticmethod
    def _normalize_inputs(
        inputs: list[MSAInputConfig] | list[tuple[float, float, str | Path]],
    ) -> list[MSAInputConfig]:
        """入力設定を標準化

        Args:
            inputs (list[MSAInputConfig] | list[tuple[float, float, str | Path]]): 入力設定のリスト

        Returns:
            list[MSAInputConfig]: 標準化された入力設定のリスト
        """
        normalized: list[MSAInputConfig] = []
        for inp in inputs:
            if isinstance(inp, MSAInputConfig):
                normalized.append(inp)  # すでに検証済みのため、そのまま追加
            else:
                fs, lag, path = inp
                normalized.append(
                    MSAInputConfig.validate_and_create(fs=fs, lag=lag, path=path)
                )
        return normalized

    @staticmethod
    def setup_logger(logger: Logger | None, log_level: int = INFO) -> Logger:
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
        new_logger: Logger = getLogger()
        # 既存のハンドラーをすべて削除
        for handler in new_logger.handlers[:]:
            new_logger.removeHandler(handler)
        new_logger.setLevel(log_level)  # ロガーのレベルを設定
        ch = StreamHandler()
        ch_formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(ch_formatter)  # フォーマッターをハンドラーに設定
        new_logger.addHandler(ch)  # StreamHandlerの追加
        return new_logger
