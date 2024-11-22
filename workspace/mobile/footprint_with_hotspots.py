import os
import pandas as pd
from matplotlib import font_manager
from omu_eddy_covariance import (
    FluxFootprintAnalyzer,
    HotspotData,
    MobileSpatialAnalyzer,
    MSAInputConfig,
)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from workspace.mobile.private.apy_key import get_api_key


# MSAInputConfigによる詳細指定
inputs: list[MSAInputConfig] = [
    MSAInputConfig(
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.10.17/input/Pico100121_241017_092120+.txt",
        delay=7,
    ),
    MSAInputConfig(
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.09/input/Pico100121_241109_103128.txt",
        delay=13,
    ),
    MSAInputConfig(
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.11/input/Pico100121_241111_091102+.txt",
        delay=13,
    ),
    MSAInputConfig(
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.14/input/Pico100121_241114_093745+.txt",
        delay=13,
    ),
    MSAInputConfig(
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.18/input/Pico100121_241118_092855+.txt",
        delay=13,
    ),
    MSAInputConfig(
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.20/input/Pico100121_241120_092932+.txt",
        delay=13,
    ),
]

num_sections: int = 4
output_dir: str = "/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private"

if __name__ == "__main__":
    analyzer = MobileSpatialAnalyzer(
        center_lat=34.573904320329724,
        center_lon=135.4829511120712,
        inputs=inputs,
        num_sections=num_sections,
        hotspot_area_meter=30,
        window_minutes=5,
        logging_debug=False,
    )

    # ホットスポット検出
    hotspots: list[HotspotData] = analyzer.analyze_hotspots(
        exclude_duplicates_across_days=True, additional_distance_meter=20
    )

    # プロジェクトルートや作業ディレクトリのパスを定義
    project_root: str = "/home/connect0459/labo/omu-eddy-covariance"
    base_path: str = f"{project_root}/workspace/footprint"
    # I/O 用ディレクトリのパス
    csv_dir_path: str = f"{base_path}/private/csv_files"
    output_dir_path: str = f"{base_path}/private/outputs"

    # ローカルフォントを読み込む場合はコメントアウトを解除して適切なパスを入力
    font_path = f"{project_root}/storage/assets/fonts/Arial/arial.ttf"
    font_manager.fontManager.addfont(font_path)

    months: list[int] = [8, 9, 10]  # 計算に含める月

    # 出力先ディレクトリを作成
    os.makedirs(output_dir_path, exist_ok=True)

    # インスタンスを作成
    analyzer_footprint = FluxFootprintAnalyzer(z_m=111, logging_debug=False)
    df: pd.DataFrame = analyzer_footprint.combine_all_csv(csv_dir_path)

    # 月ごとにデータをフィルタリング
    df = analyzer_footprint.filter_data(df, months=months)

    # 土台となる航空写真のパス
    base_image_path: str = f"{project_root}/storage/assets/images/SAC(height8000).jpg"

    # ratio
    df["Fratio"] = (df["Fc2h6 ultra"] / df["Fch4 ultra"]) / 0.076 * 100
    x_list_r, y_list_r, c_list_r = analyzer_footprint.calculate_flux_footprint(
        df, "Fratio", plot_count=30000
    )

    # 補正係数を計算（例：ある基準点について）
    lon_correction, lat_correction = analyzer_footprint.calculate_position_correction(
        reference_x=1000,  # 基準点の相対x座標(m)
        reference_y=500,  # 基準点の相対y座標(m)
        actual_lat=35.1234,  # 基準点の実際の緯度
        actual_lon=139.5678,  # 基準点の実際の経度
        center_lat=35.1200,  # 中心点の緯度
        center_lon=139.5600,  # 中心点の経度
    )

    api_key = get_api_key()

    # フットプリントとホットスポットの可視化
    analyzer_footprint.plot_flux_footprint_with_hotspots_on_api(
        x_list=x_list_r,  # メートル単位のx座標
        y_list=y_list_r,  # メートル単位のy座標
        c_list=c_list_r,
        hotspots=hotspots,
        center_lat=34.573904320329724,
        center_lon=135.4829511120712,
        api_key=api_key,
        cmap="jet",
        vmin=0,
        vmax=100,
        xy_max=4000,
        output_path=f"{output_dir}/footprint_with_hotspots-ratio.png",
    )
