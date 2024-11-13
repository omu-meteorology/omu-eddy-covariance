import os
import numpy as np
import pandas as pd
from matplotlib import font_manager
from omu_eddy_covariance import FluxFootprintAnalyzer


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

    # インスタンスを作成
    analyzer = FluxFootprintAnalyzer(z_m=111)
    df: pd.DataFrame = analyzer.combine_all_csv(csv_dir_path)

    # 月ごとにデータをフィルタリング
    months: list[int] = [9]
    df = analyzer.filter_data(df, months=months)

    # CH4
    x_list_ch4, y_list_ch4, c_list_ch4 = analyzer.calculate_flux_footprint(
        df, "Fch4 ultra"
    )
    analyzer.plot_flux_footprint(
        x_list_ch4,
        y_list_ch4,
        c_list_ch4,
        base_image_path,
        cmap="jet",
        vmin=0,
        vmax=100,
        xy_min=-4000,
        xy_max=4000,
        function=np.mean,
        cbar_label="CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
        cbar_labelpad=20,
        # output_path=f"{output_dir_path}/test-footprint-ch4.png",
        output_path=f"{output_dir_path}/footprint-ch4-9.png",
    )

    # C2H6
    x_list_c2h6, y_list_c2h6, c_list_c2h6 = analyzer.calculate_flux_footprint(
        df,
        "Fc2h6 ultra",
    )
    analyzer.plot_flux_footprint(
        x_list_c2h6,
        y_list_c2h6,
        c_list_c2h6,
        base_image_path,
        cmap="jet",
        vmin=0,
        vmax=5,
        xy_min=-4000,
        xy_max=4000,
        function=np.mean,
        cbar_label="C$_2$H$_6$ Flux (nmol m$^{-2}$ s$^{-1}$)",
        cbar_labelpad=35,
        # output_path=f"{output_dir_path}/test-footprint-c2h6.png",
        output_path=f"{output_dir_path}/footprint-c2h6-9.png",
    )

    # ratio
    # df["gas_ratio_of_Fch4"] = (df["Fc2h6 ultra"] / df["Fch4 ultra"]) / 0.076 * 100
    # x_list_ratio, y_list_ratio, c_list_ratio = analyzer.calculate_flux_footprint(
    #     df,
    #     "gas_ratio_of_Fch4",
    # )
    # analyzer.plot_flux_footprint(
    #     x_list_ratio,
    #     y_list_ratio,
    #     c_list_ratio,
    #     base_image_path,
    #     cmap="jet",
    #     vmin=0,
    #     vmax=100,
    #     xy_min=-4000,
    #     xy_max=4000,
    #     function=np.mean,
    #     cbar_label="Percentage of city gas in CH$_4$ flux (%)",
    #     cbar_labelpad=20,
    #     output_path=f"{output_dir_path}/test-footprint-ratio.png",
    # )
