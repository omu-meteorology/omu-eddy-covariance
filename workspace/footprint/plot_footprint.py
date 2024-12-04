import os
import pandas as pd
from dotenv import load_dotenv
from matplotlib import font_manager
from omu_eddy_covariance import (
    FluxFootprintAnalyzer,
    HotspotData,
    MonthlyConverter,
    MobileSpatialAnalyzer,
    MSAInputConfig,
)

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
    MSAInputConfig(
        delay=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.28/input/Pico100121_241128_090240+.txt",
    ),
    MSAInputConfig(
        delay=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.30/input/Pico100121_241130_092420+.txt",
    ),
    MSAInputConfig(
        delay=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.12.02/input/Pico100121_241202_090316+.txt",
    ),
]

# 変数定義
center_lan: float = 34.573904320329724  # 観測地点の緯度
center_lon: float = 135.4829511120712  # 観測地点の経度
num_sections: int = 4  # セクション数

# ファイルおよびディレクトリのパス
project_root: str = "/home/connect0459/labo/omu-eddy-covariance"
work_dir: str = f"{project_root}/workspace/footprint"
# I/O 用ディレクトリのパス
output_dir: str = f"{work_dir}/private/outputs"  # 出力先のディレクトリ
dotenv_path = f"{work_dir}/.env"  # .envファイル

start_date, end_date = "2024-10-01", "2024-11-30"
date_tag: str = f"-{start_date}_{end_date}"

# ローカルフォントを読み込む場合はコメントアウトを解除して適切なパスを入力
font_path = f"{project_root}/storage/assets/fonts/Arial/arial.ttf"
font_manager.fontManager.addfont(font_path)

if __name__ == "__main__":
    # 環境変数の読み込み
    load_dotenv(dotenv_path)

    # APIキーの取得
    gms_api_key: str | None = os.getenv("GOOGLE_MAPS_STATIC_API_KEY")
    if not gms_api_key:
        raise ValueError("GOOGLE_MAPS_STATIC_API_KEY is not set in .env file")

    # 出力先ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    with MonthlyConverter(
        "/home/connect0459/labo/omu-eddy-covariance/workspace/private/monthly",
        file_pattern="SA.Ultra.*.xlsx",
    ) as converter:
        # 特定の期間のデータを読み込む
        monthly_df = converter.read_sheets(
            sheet_names=["Final"], start_date=start_date, end_date=end_date
        )

    # ホットスポットの検出
    msa = MobileSpatialAnalyzer(
        center_lat=center_lan,
        center_lon=center_lon,
        inputs=inputs,
        num_sections=num_sections,
        hotspot_area_meter=50,
        window_minutes=5,
        logging_debug=False,
    )
    hotspots: list[HotspotData] = msa.analyze_hotspots(
        duplicate_check_mode="time_window"
    )

    # インスタンスを作成
    ffa = FluxFootprintAnalyzer(z_m=111, logging_debug=False)
    df: pd.DataFrame = ffa.combine_all_data(monthly_df, source_type="monthly")

    # ratio
    df["Fratio"] = (df["Fc2h6 ultra"] / df["Fch4 ultra"]) / 0.076 * 100
    x_list_r, y_list_r, c_list_r = ffa.calculate_flux_footprint(
        df=df,
        flux_key="Fratio",
        plot_count=50000,
    )

    # 航空写真の取得
    # image = ffa.get_satellite_image(
    #     api_key=gms_api_key, center_lat=center_lan, center_lon=center_lon, zoom=13
    # ) # API
    image = ffa.get_satellite_image(
        local_image_path="/home/connect0459/labo/omu-eddy-covariance/storage/assets/images/SAC-zoom_13.png"
    )  # ローカル

    # フットプリントとホットスポットの可視化
    ffa.plot_flux_footprint_with_hotspots(
        x_list=x_list_r,  # メートル単位のx座標
        y_list=y_list_r,  # メートル単位のy座標
        c_list=c_list_r,
        hotspots=hotspots,
        center_lat=center_lan,
        center_lon=center_lon,
        satellite_image=image,
        cmap="jet",
        vmin=0,
        vmax=100,
        xy_max=5000,
        cbar_label="Gas Ratio of CH$_4$ Flux (%)",
        cbar_labelpad=20,
        output_path=f"{output_dir}/footprint_ratio{date_tag}.png",
    )
