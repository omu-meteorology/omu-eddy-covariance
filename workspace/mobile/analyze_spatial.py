from omu_eddy_covariance import HotspotData, MobileSpatialAnalyzer, MSAInputConfig


# MSAInputConfigによる詳細指定
inputs: list[MSAInputConfig] = [
    MSAInputConfig(
        lag=7,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.10.17/input/Pico100121_241017_092120+.txt",
    ),
    MSAInputConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.09/input/Pico100121_241109_103128.txt",
    ),
    MSAInputConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.11/input/Pico100121_241111_091102+.txt",
    ),
    MSAInputConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.14/input/Pico100121_241114_093745+.txt",
    ),
    MSAInputConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.18/input/Pico100121_241118_092855+.txt",
    ),
    MSAInputConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.20/input/Pico100121_241120_092932+.txt",
    ),
    MSAInputConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.24/input/Pico100121_241124_092712+.txt",
    ),
    MSAInputConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.25/input/Pico100121_241125_090721+.txt",
    ),
    MSAInputConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.28/input/Pico100121_241128_090240+.txt",
    ),
    MSAInputConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.11.30/input/Pico100121_241130_092420+.txt",
    ),
    MSAInputConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/data/2024.12.02/input/Pico100121_241202_090316+.txt",
    ),
]

num_sections: int = 4
output_dir: str = "/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private"

if __name__ == "__main__":
    msa = MobileSpatialAnalyzer(
        center_lat=34.573904320329724,
        center_lon=135.4829511120712,
        inputs=inputs,
        num_sections=num_sections,
        hotspot_area_meter=50,
        window_minutes=5,
        logging_debug=False,
    )

    # msa.calculate_measurement_stats()

    # ホットスポット検出
    hotspots: list[HotspotData] = msa.analyze_hotspots(
        duplicate_check_mode="time_window",
        # duplicate_check_mode="time_all",
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
    section_size: float = msa.get_section_size()
    for section, counts in section_counts.items():
        start_angle = -180 + section * section_size
        end_angle = start_angle + section_size
        print(f"\n区画 {section} ({start_angle:.1f}° ~ {end_angle:.1f}°):")
        print(f"  Bio  : {counts['bio']}")
        print(f"  Gas  : {counts['gas']}")
        print(f"  Comb : {counts['comb']}")

    # 地図の作成と保存
    msa.create_hotspots_map(hotspots, output_dir=output_dir)

    # ホットスポットを散布図で表示
    msa.plot_scatter_c2h6_ch4(output_dir=output_dir)

    # ヒストグラムを作図
    msa.plot_ch4_delta_histogram(
        output_dir=output_dir,
        figsize=(8, 6),
        xlim=(0, 2),
    )
