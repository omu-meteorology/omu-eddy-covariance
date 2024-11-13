import os
import pandas as pd
from matplotlib import font_manager
from omu_eddy_covariance import (
    GeoFluxFootprintAnalyzer,
)

# プロジェクトルートや作業ディレクトリのパスを定義
project_root: str = "/home/connect0459/workspace/labo/omu-eddy-covariance"
base_path: str = f"{project_root}/workspace/footprint"
# I/O 用ディレクトリのパス
csv_dir_path: str = f"{base_path}/private/csv_files"
output_dir_path: str = f"{base_path}/private/outputs"

# 土台となる航空写真のパス
base_image_path: str = f"{project_root}/storage/assets/images/SAC(height8000).jpg"

# ローカルフォントを読み込む場合はコメントアウトを解除して適切なパスを入力
font_path = f"{project_root}/storage/assets/fonts/Arial/arial.ttf"
font_manager.fontManager.addfont(font_path)

if __name__ == "__main__":
    # 出力先ディレクトリを作成
    os.makedirs(output_dir_path, exist_ok=True)

    # 初期化
    analyzer = GeoFluxFootprintAnalyzer(
        tower_lat=34.57397868845166,  # タワーの緯度
        tower_lon=135.48288773915024,  # タワーの経度
        z_m=111,  # 測定高度
    )

    df: pd.DataFrame = analyzer.combine_all_csv(csv_dir_path)

    # 月ごとにデータをフィルタリング
    months: list[int] = [7]
    df = analyzer.filter_data(df, months=months)

    # フットプリントの計算（既存のメソッド）
    x_list, y_list, c_list = analyzer.calculate_flux_footprint(
        df, "Fch4 ultra", plot_count=30000
    )

    # インタラクティブ地図の生成
    map_obj = analyzer.plot_geographic_footprint(
        x_list,
        y_list,
        c_list,
        zoom_start=15,
        radius=1000,  # 1km円を表示
        output_path=f"{output_dir_path}/footprint_interactive.html",
    )

    # 静的地図の生成
    fig = analyzer.plot_static_geographic_footprint(
        x_list,
        y_list,
        c_list,
        radius=1000,
        output_path=f"{output_dir_path}/footprint_static.png",
    )
