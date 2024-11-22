import os
import numpy as np
import pandas as pd
from matplotlib import font_manager
from omu_eddy_covariance import FluxFootprintAnalyzer
from omu_eddy_covariance import HotspotData, MobileSpatialAnalyzer, MSAInputConfig

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

    # 結果の表示
    bio_spots = [h for h in hotspots if h.type == "bio"]
    gas_spots = [h for h in hotspots if h.type == "gas"]
    comb_spots = [h for h in hotspots if h.type == "comb"]

    print("\nResults:")
    print(f"  Bio:{len(bio_spots)},Gas:{len(gas_spots)},Comb:{len(comb_spots)}")

    # 区画ごとの分析を追加
    # 各区画のホットスポット数をカウント
    section_counts = {i: {"bio": 0, "gas": 0, "comb": 0} for i in range(num_sections)}
    for spot in hotspots:
        section_counts[spot.section][spot.type] += 1

    # 区画ごとの結果を表示
    print("\n区画ごとの分析結果:")
    section_size: float = analyzer.get_section_size()
    for section, counts in section_counts.items():
        start_angle = -180 + section * section_size
        end_angle = start_angle + section_size
        print(f"\n区画 {section} ({start_angle:.1f}° ~ {end_angle:.1f}°):")
        print(f"  Bio  : {counts['bio']}")
        print(f"  Gas  : {counts['gas']}")
        print(f"  Comb : {counts['comb']}")

    # # 地図の作成と保存
    # analyzer.create_hotspots_map(hotspots, output_dir=output_dir)

    # # ホットスポットを散布図で表示
    # analyzer.plot_scatter_c2h6_ch4(output_dir=output_dir)

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
    analyzer_footprint = FluxFootprintAnalyzer(z_m=111)
    df: pd.DataFrame = analyzer_footprint.combine_all_csv(csv_dir_path)

    # 月ごとにデータをフィルタリング
    df = analyzer_footprint.filter_data(df, months=months)

    # 土台となる航空写真のパス
    base_image_path: str = f"{project_root}/storage/assets/images/SAC(height8000).jpg"

    # # CH4
    # x_list_ch4, y_list_ch4, c_list_ch4 = analyzer_footprint.calculate_flux_footprint(
    #     df, "Fch4 ultra", plot_count=50000
    # )
    # # フットプリントとホットスポットの可視化
    # analyzer_footprint.plot_flux_footprint_with_hotspots(
    #     x_list=x_list_ch4,
    #     y_list=y_list_ch4,
    #     c_list=c_list_ch4,
    #     center_lat=34.573904320329724,
    #     center_lon=135.4829511120712,
    #     hotspots=hotspots,
    #     base_image_path=base_image_path,
    #     cmap="jet",
    #     vmin=0,
    #     vmax=100,
    #     xy_min=-4000,
    #     xy_max=4000,
    #     function=np.mean,
    #     cbar_label="CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
    #     cbar_labelpad=20,
    #     output_path=f"{output_dir}/footprint_with_hotspots-ch4.png",
    # )

    # ratio
    df["Fratio"] = (df["Fc2h6 ultra"] / df["Fch4 ultra"]) / 0.076 * 100
    x_list_r, y_list_r, c_list_r = analyzer_footprint.calculate_flux_footprint(
        df, "Fratio", plot_count=30000
    )
    # フットプリントとホットスポットの可視化
    # analyzer_footprint.plot_flux_footprint_with_hotspots(
    #     x_list=x_list_r,
    #     y_list=y_list_r,
    #     c_list=c_list_r,
    #     center_lat=34.573904320329724,
    #     center_lon=135.4829511120712,
    #     hotspots=hotspots,
    #     base_image_path=base_image_path,
    #     cmap="jet",
    #     vmin=0,
    #     vmax=100,
    #     xy_min=-4000,
    #     xy_max=4000,
    #     function=np.mean,
    #     cbar_label="Gas Ratio of CH$_4$ Flux (%)",
    #     cbar_labelpad=20,
    #     output_path=f"{output_dir}/footprint_with_hotspots-ratio.png",
    # )

    api_key = get_api_key()

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
        xy_min=-4000,  # メートル単位の表示範囲
        xy_max=4000,
        output_path=f"{output_dir}/footprint_with_hotspots-ratio.png",
        auto_zoom=True,
    )
