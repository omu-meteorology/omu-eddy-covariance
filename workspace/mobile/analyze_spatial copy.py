import os
import pandas as pd
from matplotlib import font_manager
from omu_eddy_covariance import FluxFootprintAnalyzer
from omu_eddy_covariance import HotspotData, MobileSpatialAnalyzer, MSAInputConfig


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

    months: list[int] = [6, 7, 8]  # 計算に含める月
    tag: str = "6_8"  # 画像ファイルに付与するタグ

    # 出力先ディレクトリを作成
    os.makedirs(output_dir_path, exist_ok=True)

    # インスタンスを作成
    analyzer_footprint = FluxFootprintAnalyzer(z_m=111)
    df: pd.DataFrame = analyzer_footprint.combine_all_csv(csv_dir_path)

    # 月ごとにデータをフィルタリング
    df = analyzer_footprint.filter_data(df, months=months)

    # CH4
    x_list_ch4, y_list_ch4, c_list_ch4 = analyzer_footprint.calculate_flux_footprint(
        df, "Fch4 ultra", plot_count=10000
    )

    # マップを作成
    analyzer.create_footprint_map(
        x_list=x_list_ch4,
        y_list=y_list_ch4,
        c_list=c_list_ch4,
        hotspots=hotspots,
        output_dir=output_dir,
    )
    