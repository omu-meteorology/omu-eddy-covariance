import folium
from folium import plugins
import pandas as pd
from pyproj import Transformer
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from logging import Logger
from .flux_footprint_analyzer import FluxFootprintAnalyzer


class GeoFluxFootprintAnalyzer(FluxFootprintAnalyzer):
    """
    地理座標系に対応したフラックスフットプリントアナライザー

    このクラスは既存のFluxFootprintAnalyzerを拡張し、
    フットプリントを地理座標系（緯度経度）で表示する機能を追加します。
    """

    def __init__(
        self,
        tower_lat: float,
        tower_lon: float,
        z_m: float,
        labelsize: float = 20,
        ticksize: float = 16,
        plot_params=None,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        タワーの位置情報を含めて初期化します。

        Args:
            tower_lat (float): タワーの緯度
            tower_lon (float): タワーの経度
            z_m (float): 測定の高さ（メートル単位）。
            labelsize (float): 軸ラベルのフォントサイズ。デフォルトは20。
            ticksize (float): 軸目盛りのフォントサイズ。デフォルトは16。
            plot_params (Optional[Dict[str, any]]): matplotlibのプロットパラメータを指定する辞書。
            logger (Logger | None): 使用するロガー。Noneの場合は新しいロガーを生成します。
            logging_debug (bool): ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。
        """
        super().__init__(
            z_m=z_m,
            labelsize=labelsize,
            ticksize=ticksize,
            plot_params=plot_params,
            logger=logger,
            logging_debug=logging_debug,
        )

        self.tower_lat = tower_lat
        self.tower_lon = tower_lon

        # 投影変換用のTransformerを初期化
        # WGS84からUTM座標系への変換器
        zone_number = int((tower_lon + 180) / 6) + 1
        self.proj_string = (
            f"+proj=utm +zone={zone_number} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        )
        self.transformer = Transformer.from_crs(
            "EPSG:4326",  # WGS84 (緯度経度)
            self.proj_string,  # UTM
            always_xy=True,
        )

        # 逆変換用のTransformer
        self.reverse_transformer = Transformer.from_crs(
            self.proj_string,  # UTM
            "EPSG:4326",  # WGS84
            always_xy=True,
        )

    def xy_to_latlon(self, x: float, y: float) -> tuple[float, float]:
        """
        相対座標（メートル）を緯度経度に変換します。

        Args:
            x (float): タワーからの東西距離（メートル）
            y (float): タワーからの南北距離（メートル）

        Returns:
            tuple[float, float]: (緯度, 経度)
        """
        # タワー位置をUTM座標に変換
        tower_x, tower_y = self.transformer.transform(self.tower_lon, self.tower_lat)

        # 相対座標を絶対UTM座標に変換
        abs_x = tower_x + x
        abs_y = tower_y + y

        # UTM座標を緯度経度に変換
        lon, lat = self.reverse_transformer.transform(abs_x, abs_y)
        return lat, lon

    def plot_geographic_footprint(
        self,
        x_list: list[float],
        y_list: list[float],
        c_list: list[float],
        zoom_start: int = 15,
        radius: int = 1000,
        tiles: str = "OpenStreetMap",
        output_path: str = "",
    ) -> folium.Map:
        """
        フットプリントを地理座標系で可視化し、インタラクティブな地図を生成します。

        Args:
            x_list (list[float]): x座標のリスト（メートル）
            y_list (list[float]): y座標のリスト（メートル）
            c_list (list[float]): フラックス値のリスト
            zoom_start (int): 初期表示時のズームレベル
            radius (int): 表示範囲の半径（メートル）
            tiles (str): 背景地図のタイプ
            cmap (str): カラーマップ名
            output_path (str): 出力ファイルパス（.html）

        Returns:
            folium.Map: 生成された地図オブジェクト
        """
        # データをDataFrameに変換
        df = pd.DataFrame({"x": x_list, "y": y_list, "flux": c_list})

        # 座標変換
        df["lat"], df["lon"] = zip(
            *[self.xy_to_latlon(x, y) for x, y in zip(df["x"], df["y"])]
        )

        self.logger.info("地図を作成します。")

        # 地図の初期化
        m = folium.Map(
            location=[self.tower_lat, self.tower_lon],
            zoom_start=zoom_start,
            tiles=tiles,
        )

        # タワー位置にマーカーを追加
        folium.Marker(
            [self.tower_lat, self.tower_lon],
            popup="Tower",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

        # フットプリントのヒートマップを作成
        data = df[["lat", "lon", "flux"]].values.tolist()
        plugins.HeatMap(
            data,
            radius=15,
            blur=10,
            gradient={0.4: "blue", 0.6: "lime", 0.8: "yellow", 1: "red"},
        ).add_to(m)

        # 円を追加してスケールを視覚化
        folium.Circle(
            location=[self.tower_lat, self.tower_lon],
            radius=radius,
            color="red",
            fill=False,
            popup=f"{radius}m",
        ).add_to(m)

        self.logger.info("地図の作成が完了しました。")

        # 保存
        if output_path:
            self.logger.info("地図を保存しています...")
            m.save(output_path)
            self.logger.info("地図の保存が完了しました。")

        return m

    def plot_static_geographic_footprint(
        self,
        x_list: list[float],
        y_list: list[float],
        c_list: list[float],
        radius: int = 1000,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        output_path: str = "",
    ) -> plt.Figure:
        """
        フットプリントを地理座標系で静的な地図として可視化します。

        Args:
            x_list (list[float]): x座標のリスト（メートル）
            y_list (list[float]): y座標のリスト（メートル）
            c_list (list[float]): フラックス値のリスト
            radius (int): 表示範囲の半径（メートル）
            cmap (str): カラーマップ名
            vmin (float | None): カラースケールの最小値
            vmax (float | None): カラースケールの最大値
            output_path (str): 出力ファイルパス

        Returns:
            plt.Figure: 生成された図のオブジェクト
        """
        # データをDataFrameに変換し、GeoDataFrameを作成
        df = pd.DataFrame({"x": x_list, "y": y_list, "flux": c_list})

        # 座標変換
        df["lat"], df["lon"] = zip(
            *[self.xy_to_latlon(x, y) for x, y in zip(df["x"], df["y"])]
        )

        # GeoDataFrameの作成
        gdf = gpd.GeoDataFrame(
            df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs="EPSG:4326"
        )

        # UTM座標系に変換
        gdf = gdf.to_crs(self.proj_string)

        self.logger.info("地図画像を作成します。")

        # プロットの作成
        fig, ax = plt.subplots(figsize=(12, 12))

        # ヘキサゴンビンでフットプリントを描画
        hb = ax.hexbin(
            gdf.geometry.x,
            gdf.geometry.y,
            C=gdf["flux"],
            gridsize=50,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.6,
        )

        # 背景地図の追加
        ctx.add_basemap(
            ax, crs=self.proj_string, source=ctx.providers.OpenStreetMap.Mapnik
        )

        # タワー位置に印をつける
        tower_x, tower_y = self.transformer.transform(self.tower_lon, self.tower_lat)
        ax.plot(tower_x, tower_y, "r^", markersize=10, label="Tower")

        # 半径円の追加
        circle = plt.Circle(
            (tower_x, tower_y),
            radius,
            fill=False,
            color="red",
            linestyle="--",
            label=f"{radius}m",
        )
        ax.add_patch(circle)

        # 凡例とカラーバーの追加
        plt.colorbar(hb, ax=ax, label="Flux")
        ax.legend()

        # 軸の設定
        ax.set_aspect("equal")

        self.logger.info("地図画像の作成が完了しました。")

        # 保存
        if output_path:
            self.logger.info("地図画像を保存しています...")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            self.logger.info("地図画像の保存が完了しました。")

        return fig
