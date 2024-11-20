from omu_eddy_covariance import HotspotData, MobileSpatialAnalyzer, MSAInputConfig

# 設定例：MSAInputConfigによる詳細指定
inputs = [
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

analyzer = MobileSpatialAnalyzer(
    center_lat=34.57397868845166,
    center_lon=135.48288773915024,
    inputs=inputs,
    num_sections=4,
    sampling_frequency=1,
    window_minutes=5.0,
    logging_debug=False,
)

# ホットスポット検出
hotspots: list[HotspotData] = analyzer.analyze_hotspots()

# 結果の表示
comb_spots = [h for h in hotspots if h.type == "comb"]
gas_spots = [h for h in hotspots if h.type == "gas"]
bio_spots = [h for h in hotspots if h.type == "bio"]

print("\nResults:")
print(
    f"Comb hotspots:{len(comb_spots)},Gas hotspots:{len(gas_spots)},Bio hotspots:{len(bio_spots)}"
)

# 区画ごとの分析を追加
# 各区画のホットスポット数をカウント
section_counts = {
    i: {"comb": 0, "gas": 0, "bio": 0} for i in range(analyzer.num_sections)
}
for spot in hotspots:
    section_counts[spot.section][spot.type] += 1

# 区画ごとの結果を表示
print("\n区画ごとの分析結果:")
for section, counts in section_counts.items():
    start_angle = -180 + section * analyzer.section_size
    end_angle = start_angle + analyzer.section_size
    print(f"\n区画 {section} ({start_angle:.1f}° ~ {end_angle:.1f}°):")
    print(f"  Comb : {counts['comb']}")
    print(f"  Gas  : {counts['gas']}")
    print(f"  Bio  : {counts['bio']}")

# ホットスポットの詳細情報
# for spot in hotspots:
#     print("\nHotspot:")
#     print(f"Type: {spot.type}")
#     print(f"Location: ({spot.avg_lat}, {spot.avg_lon})")
#     print(f"C2H6/CH4 ratio: {spot.ratio:.2f}")

# 地図の作成と保存
output_path = "/home/connect0459/labo/omu-eddy-covariance/workspace/mobile/private/hotspots_map.html"
analyzer.create_hotspots_map(hotspots, output_path)
