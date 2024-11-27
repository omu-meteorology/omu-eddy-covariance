import os
import pandas as pd
from dotenv import load_dotenv
from matplotlib import font_manager
from omu_eddy_covariance import (
    FluxFootprintAnalyzer,
    HotspotData,
    MobileSpatialAnalyzer,
    MSAInputConfig,
)

# 変数定義
center_lan: float = 34.573904320329724  # 観測地点の緯度
center_lon: float = 135.4829511120712  # 観測地点の経度
num_sections: int = 4  # セクション数
months: list[int] = [8, 9, 10]  # 計算に含める月

# ファイルおよびディレクトリのパス
project_root: str = "/home/connect0459/labo/omu-eddy-covariance"
work_dir: str = f"{project_root}/workspace/footprint"
# I/O 用ディレクトリのパス
csv_dir: str = f"{work_dir}/private/csv_files"  # csvファイルの入ったディレクトリ
output_dir: str = f"{work_dir}/private/outputs"  # 出力先のディレクトリ
dotenv_path = f"{work_dir}/.env"  # .envファイル

# ローカルフォントを読み込む場合はコメントアウトを解除して適切なパスを入力
font_path = f"{project_root}/storage/assets/fonts/Arial/arial.ttf"
font_manager.fontManager.addfont(font_path)

# MSAInputConfigによる詳細指定
inputs: list[MSAInputConfig] = [
    MSAInputConfig(
        delay=7,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.10.17/input/Pico100121_241017_092120+.txt",
    ),
    MSAInputConfig(
        delay=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.09/input/Pico100121_241109_103128.txt",
    ),
    MSAInputConfig(
        delay=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.11/input/Pico100121_241111_091102+.txt",
    ),
    MSAInputConfig(
        delay=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.14/input/Pico100121_241114_093745+.txt",
    ),
    MSAInputConfig(
        delay=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.18/input/Pico100121_241118_092855+.txt",
    ),
    MSAInputConfig(
        delay=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.20/input/Pico100121_241120_092932+.txt",
    ),
    MSAInputConfig(
        delay=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.24/input/Pico100121_241124_092712+.txt",
    ),
    MSAInputConfig(
        delay=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.25/input/Pico100121_241125_090721+.txt",
    ),
]

if __name__ == "__main__":
    # 環境変数の読み込み
    load_dotenv(dotenv_path)

    # APIキーの取得
    gms_api_key: str | None = os.getenv("GOOGLE_MAPS_STATIC_API_KEY")
    if not gms_api_key:
        raise ValueError("GOOGLE_MAPS_STATIC_API_KEY is not set in .env file")

    # 出力先ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # ホットスポットの検出
    analyzer = MobileSpatialAnalyzer(
        center_lat=center_lan,
        center_lon=center_lon,
        inputs=inputs,
        num_sections=num_sections,
        hotspot_area_meter=30,
        window_minutes=5,
        logging_debug=False,
    )-[]
    hotspots: list[HotspotData] = analyzer.analyze_hotspots(
        exclude_duplicates_across_days=True, additional_distance_meter=20
    )

    # インスタンスを作成
    analyzer_footprint = FluxFootprintAnalyzer(z_m=111, logging_debug=False)
    df: pd.DataFrame = analyzer_footprint.combine_all_csv(csv_dir)

    # 月ごとにデータをフィルタリング
    df = analyzer_footprint.filter_data(df, months=months)

    # ratio
    df["Fratio"] = (df["Fc2h6 ultra"] / df["Fch4 ultra"]) / 0.076 * 100
    x_list_r, y_list_r, c_list_r = analyzer_footprint.calculate_flux_footprint(
        # df, "Fratio", plot_count=30000
        df,
        "Fratio",
        plot_count=50000,
    )

    # フットプリントとホットスポットの可視化
    analyzer_footprint.plot_flux_footprint_with_hotspots_on_api(
        x_list=x_list_r,  # メートル単位のx座標
        y_list=y_list_r,  # メートル単位のy座標
        c_list=c_list_r,
        hotspots=hotspots,
        center_lat=center_lan,
        center_lon=center_lon,
        api_key=gms_api_key,
        cmap="jet",
        vmin=0,
        vmax=100,
        xy_max=4000,
        output_path=f"{output_dir}/footprint_with_hotspots-ratio.png",
    )

    # フットプリントとホットスポットの可視化
    analyzer_footprint.plot_flux_footprint_with_hotspots_on_api(
        x_list=x_list_r,  # メートル単位のx座標
        y_list=y_list_r,  # メートル単位のy座標
        c_list=c_list_r,
        hotspots=hotspots,
        center_lat=center_lan,
        center_lon=center_lon,
        api_key=gms_api_key,
        cmap="jet",
        vmin=0,
        vmax=100,
        xy_max=4000,
        output_path=f"{output_dir}/footprint_with_hotspots-ratio.png",
    )
