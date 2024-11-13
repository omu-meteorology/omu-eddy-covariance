from omu_eddy_covariance import MobileSpatialAnalyzer, MSAInputConfig

# 設定例：MSAInputConfigによる詳細指定
inputs = [
    # MSAInputConfig(
    #     path="/home/connect0459/workspace/labo/omu-meteorology/workspace/mobile/private/data/2024.10.17/input/Pico100121_241017_092120+.txt",
    #     delay=7,
    # ),
    # MSAInputConfig(
    #     path="/home/connect0459/workspace/labo/omu-meteorology/workspace/mobile/private/data/2024.11.09/input/Pico100121_241109_103128.txt",
    #     delay=7,
    # ),
    MSAInputConfig(
        path="/home/connect0459/workspace/labo/omu-meteorology/workspace/mobile/private/data/2024.11.11/input/Pico100121_241111_091102+.txt",
        delay=7,
    ),
]

analyzer = MobileSpatialAnalyzer(
    center_lat=34.57397868845166,
    center_lon=135.48288773915024,
    inputs=inputs,
    num_sections=4,
    sampling_frequency=1,
    window_minutes=5.0,
    logging_debug=True,
)

# ホットスポット検出
hotspots = analyzer.analyze_hotspots()

# 結果の表示
bio_spots = [h for h in hotspots if h.type == "bio"]
gas_spots = [h for h in hotspots if h.type == "gas"]

print("\nResults:")
print(f"Bio hotspots: {len(bio_spots)}")
print(f"Gas hotspots: {len(gas_spots)}")

# ホットスポットの詳細情報
for spot in hotspots:
    print("\nHotspot:")
    print(f"Type: {spot.type}")
    print(f"Location: ({spot.avg_lat}, {spot.avg_lon})")
    print(f"C2H6/CH4 ratio: {spot.ratio:.2f}")
