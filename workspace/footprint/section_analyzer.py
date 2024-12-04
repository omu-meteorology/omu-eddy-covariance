import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from dataclasses import dataclass
from omu_eddy_covariance import (
    HotspotData,
    FluxFootprintAnalyzer,
    MobileSpatialAnalyzer,
    MonthlyConverter,
    MSAInputConfig,
)


@dataclass
class SectorAnalysisResult:
    """セクター分析の結果を格納するデータクラス"""

    section: int
    footprint_intensity: float  # フットプリント強度の平均値
    hotspot_count: int  # ホットスポット数
    p_value: float  # 統計的有意性のp値
    correlation: float  # 相関係数


class FluxSpatialAnalyzer:
    def __init__(self, num_sections: int = 8):
        """
        空間解析クラスの初期化

        Args:
            num_sections (int): 分析する区画数（デフォルトは8で、45度ごとの区画）
        """
        self.num_sections = num_sections
        self.section_size = 360 / num_sections

    def analyze_section_correlation(
        self,
        footprint_x: list[float],
        footprint_y: list[float],
        footprint_values: list[float],
        hotspots: list[HotspotData],
    ) -> list[SectorAnalysisResult]:
        """
        セクターごとのフットプリントとホットスポットの関連性を分析

        Args:
            footprint_x: フットプリントのx座標リスト
            footprint_y: フットプリントのy座標リスト
            footprint_values: フットプリント強度値のリスト
            hotspots: ホットスポットデータのリスト

        Returns:
            list[SectorAnalysisResult]: 各セクターの分析結果
        """
        # セクターごとの結果を格納するリスト
        results = []

        # フットプリントデータをセクターごとに分類
        footprint_sections = self.__classify_footprint_to_sections(
            footprint_x, footprint_y, footprint_values
        )

        # ホットスポットデータをセクターごとに分類
        hotspot_sections = self.__classify_hotspots_to_sections(hotspots)

        # 各セクターで分析を実行
        for section in range(self.num_sections):
            # セクターごとのフットプリント強度を計算
            section_footprint = footprint_sections.get(section, [])
            mean_intensity = np.mean(section_footprint) if section_footprint else 0

            # セクターごとのホットスポット数をカウント
            hotspot_count = len(hotspot_sections.get(section, []))

            # 統計的有意性の検定
            if section_footprint and hotspot_count > 0:
                # フットプリント強度とホットスポット位置の相関を計算
                correlation, p_value = self.__calculate_spatial_correlation(
                    section_footprint, hotspot_sections.get(section, [])
                )
            else:
                correlation, p_value = 0, 1.0

            results.append(
                SectorAnalysisResult(
                    section=section,
                    footprint_intensity=mean_intensity,
                    hotspot_count=hotspot_count,
                    p_value=p_value,
                    correlation=correlation,
                )
            )

        return results

    def run_analysis(
        self,
        x_list: list[float],
        y_list: list[float],
        c_list: list[float],
        hotspots: list[HotspotData],
        num_sections: int = 8,
    ) -> list[SectorAnalysisResult]:
        """
        フットプリントとホットスポットの空間的関連性の分析を実行

        Args:
            x_list: フットプリントのx座標リスト
            y_list: フットプリントのy座標リスト
            c_list: フットプリント強度値のリスト
            hotspots: ホットスポットデータのリスト
            num_sections: 分析する区画数

        Returns:
            list[SectorAnalysisResult]: 分析結果のリスト
        """
        analyzer = FluxSpatialAnalyzer(num_sections=num_sections)
        results = analyzer.analyze_section_correlation(
            x_list,
            y_list,
            c_list,
            hotspots,
        )

        # 結果の統計的解釈
        significant_sections = [r for r in results if r.p_value < 0.05]
        print(
            f"\n有意な相関が見られたセクター数: {len(significant_sections)}/{num_sections}"
        )

        for r in sorted(results, key=lambda x: x.p_value):
            print(f"\nセクター {r.section}:")
            print(f"  フットプリント強度: {r.footprint_intensity:.4f}")
            print(f"  ホットスポット数: {r.hotspot_count}")
            print(f"  相関係数: {r.correlation:.4f}")
            print(f"  p値: {r.p_value:.4f}")

        return results

    def visualize_results(
        self,
        results: list[SectorAnalysisResult],
        title: str = "Sector Analysis Results",
    ) -> plt.Figure:
        """分析結果の可視化（セクター0-3のみ）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # セクター0-3のデータのみを抽出
        filtered_results = [r for r in results if r.section <= 3]

        sections = [r.section for r in filtered_results]
        intensities = [r.footprint_intensity for r in filtered_results]
        counts = [r.hotspot_count for r in filtered_results]
        p_values = [r.p_value for r in filtered_results]

        # フットプリント強度とホットスポット数の比較
        ax1.bar(sections, intensities, alpha=0.5, label="Footprint Intensity")
        ax1_twin = ax1.twinx()
        ax1_twin.plot(sections, counts, "r-", marker="o", label="Hotspot Count")

        ax1.set_xlabel("Sector")
        ax1.set_ylabel("Footprint Intensity")
        ax1_twin.set_ylabel("Hotspot Count")
        ax1.legend(loc="upper left")
        ax1_twin.legend(loc="upper right")

        # p値の可視化
        significance_levels = [0.01, 0.05, 0.1]
        colors = ["darkgreen", "green", "lightgreen"]

        ax2.bar(sections, [-np.log10(p) for p in p_values])
        for level, color in zip(significance_levels, colors):
            ax2.axhline(
                -np.log10(level), color=color, linestyle="--", label=f"p={level}"
            )

        ax2.set_xlabel("Sector")
        ax2.set_ylabel("-log10(p-value)")
        ax2.legend()

        ax1.set_xlim(-0.5, 3.5)
        ax1.set_xticks(range(4))  # 0,1,2,3のみ表示

        ax2.set_xlim(-0.5, 3.5)
        ax2.set_xticks(range(4))  # 0,1,2,3のみ表示

        plt.tight_layout()
        return fig

    def visualize_combined_results(
        self,
        results_bio: list[SectorAnalysisResult],
        results_gas: list[SectorAnalysisResult],
        title: str = "Combined Analysis Results",
        flux_type: str = "CH4 flux",  # または "C2H6/CH4 ratio"
    ) -> plt.Figure:
        """bioとgasの結果を組み合わせて表示"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # セクター0-3のデータのみを抽出
        filtered_bio = [r for r in results_bio if r.section <= 3]
        filtered_gas = [r for r in results_gas if r.section <= 3]

        sections = [r.section for r in filtered_bio]
        intensities = [r.footprint_intensity for r in filtered_bio]  # bioとgasで同じ
        bio_counts = [r.hotspot_count for r in filtered_bio]
        gas_counts = [r.hotspot_count for r in filtered_gas]
        bio_p_values = [r.p_value for r in filtered_bio]
        gas_p_values = [r.p_value for r in filtered_gas]

        # 左パネル：Footprint IntensityとHotspot Counts
        ax1.bar(
            sections, intensities, alpha=0.5, color="lightblue", label=f"{flux_type}"
        )
        ax1_twin = ax1.twinx()

        # bioとgasのホットスポット数をプロット
        ax1_twin.plot(sections, bio_counts, "b-", marker="o", label="Bio Hotspots")
        ax1_twin.plot(sections, gas_counts, "r-", marker="s", label="Gas Hotspots")

        ax1.set_xlabel("Sector")

        # 軸ラベルの単位を追加
        if flux_type == "CH4 flux":
            flux_label = r"CH$_4$ Flux"
            ax1.set_ylabel(f"{flux_label} (nmol m$^{{-2}}$ s$^{{-1}}$)")
        else:
            flux_label = "C$_2$H$_6$/CH$_4$ Flux Ratio"
            ax1.set_ylabel(f"{flux_label} Intensity (%)")  # C2H6/CH4比の場合

        ax1_twin.set_ylabel("Hotspot Count")

        # 2つのy軸の凡例を結合
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # 右パネル：p値の比較
        significance_levels = [0.01, 0.05, 0.1]
        colors = ["darkgreen", "green", "lightgreen"]

        width = 0.35  # バーの幅
        bio_positions = np.array(sections) - width / 2
        gas_positions = np.array(sections) + width / 2

        # bioとgasのp値を並べて表示
        ax2.bar(
            bio_positions,
            [-np.log10(p) for p in bio_p_values],
            width,
            label="Bio",
            color="blue",
            alpha=0.6,
        )
        ax2.bar(
            gas_positions,
            [-np.log10(p) for p in gas_p_values],
            width,
            label="Gas",
            color="red",
            alpha=0.6,
        )

        for level, color in zip(significance_levels, colors):
            ax2.axhline(
                -np.log10(level), color=color, linestyle="--", label=f"p={level}"
            )

        ax2.set_xlabel("Sector")
        ax2.set_ylabel("-log10(p-value)")
        ax2.legend()

        ax1.set_xlim(-0.5, 3.5)
        ax1.set_xticks(range(4))  # 0,1,2,3のみ表示

        ax2.set_xlim(-0.5, 3.5)
        ax2.set_xticks(range(4))  # 0,1,2,3のみ表示

        plt.suptitle(title)
        plt.tight_layout()
        return fig

    # def __calculate_spatial_correlation(
    #     self, footprint_values: list[float], hotspots: list[HotspotData]
    # ) -> tuple[float, float]:
    #     """
    #     フットプリント強度とホットスポット分布の空間相関を計算

    #     Returns:
    #         tuple[float, float]: (相関係数, p値)
    #     """
    #     if not footprint_values or not hotspots:
    #         return 0, 1.0

    #     # フットプリント強度の分布を正規化
    #     normalized_footprint = np.array(footprint_values)
    #     normalized_footprint = (
    #         normalized_footprint - np.mean(normalized_footprint)
    #     ) / np.std(normalized_footprint)

    #     # ホットスポットの分布を二値化配列として作成
    #     hotspot_distribution = np.zeros(len(footprint_values))
    #     for i in range(len(hotspots)):
    #         hotspot_distribution[i] = 1

    #     # スピアマンの順位相関係数を計算
    #     correlation, p_value = stats.spearmanr(
    #         normalized_footprint, hotspot_distribution
    #     )

    #     return correlation, p_value
    
    def _calculate_spatial_correlation(
        self, footprint_values: list[float], hotspots: list[HotspotData]
    ) -> tuple[float, float]:
        """
        フットプリント強度とホットスポット分布の空間相関を計算
        
        Args:
            footprint_values: セクター内のフットプリント強度値
            hotspots: セクター内のホットスポットデータ
            
        Returns:
            Tuple[float, float]: (相関係数, p値)
        """
        if not footprint_values or not hotspots:
            return 0, 1.0
            
        # セクターを等間隔のグリッドに分割
        grid_size = 10  # グリッドの数
        
        # グリッドごとの統計量を計算
        grid_footprint = np.zeros(grid_size)
        grid_hotspot = np.zeros(grid_size)
        
        # フットプリント値をグリッドに割り当て
        for i, value in enumerate(footprint_values):
            grid_index = i % grid_size  # 単純な例として
            grid_footprint[grid_index] += value
            
        # ホットスポットをグリッドに割り当て
        for spot in hotspots:
            # 実際の実装ではホットスポットの座標を使用してグリッドを決定
            grid_index = hash(str(spot)) % grid_size  # 単純な例として
            grid_hotspot[grid_index] += 1
        
        # グリッドベースでスピアマンの順位相関を計算
        correlation, p_value = stats.spearmanr(grid_footprint, grid_hotspot)
        
        return correlation, p_value

    def __classify_footprint_to_sections(
        self, x: list[float], y: list[float], values: list[float]
    ) -> dict[int, list[float]]:
        """
        フットプリントデータをセクターごとに分類（-180°～180°）

        Args:
            x: 東方向の座標（正が東）
            y: 北方向の座標（正が北）
            values: フラックス値

        Returns:
            dict[int, list[float]]: セクター番号をキーとするフラックス値のリスト

        Notes:
            - 角度は-180°から時計回りに計算
            - 北を0°とし、東が90°、南が±180°、西が-90°
            - セクター0は-180°から開始
        """
        sections = {}

        for i in range(len(x)):
            # 北を0°として反時計回りに角度を計算（arctan2の標準）
            angle = np.degrees(np.arctan2(x[i], y[i]))

            # -180°～180°の範囲になるよう調整（すでにこの範囲）

            # セクター番号を計算（-180°が0番）
            # 例：8セクターの場合、各セクター45°
            # -180° -> 0
            # -135° -> 1
            # -90°  -> 2
            # -45°  -> 3
            # 0°    -> 4
            # 45°   -> 5
            # 90°   -> 6
            # 135°  -> 7
            section = int((angle + 180) / self.section_size)

            # 360°でラップアラウンドする場合の処理
            if section >= self.num_sections:
                section = 0

            # セクターにデータを追加
            if section not in sections:
                sections[section] = []
            sections[section].append(values[i])

        return sections

    def __classify_hotspots_to_sections(
        self, hotspots: list[HotspotData]
    ) -> dict[int, list[HotspotData]]:
        """
        ホットスポットをセクターごとに分類

        Args:
            hotspots: ホットスポットデータのリスト

        Returns:
            dict[int, list[HotspotData]]: セクター番号をキーとするホットスポットのリスト

        Notes:
            - HotspotDataのsectionは既に-180°を起点とした区画番号として計算されていると仮定
        """
        sections = {}

        for spot in hotspots:
            # 既存のsection値を使用
            section = spot.section

            # セクターにデータを追加
            if section not in sections:
                sections[section] = []
            sections[section].append(spot)

        return sections


"""
変数定義
"""
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

center_lan: float = 34.573904320329724  # 観測地点の緯度
center_lon: float = 135.4829511120712  # 観測地点の経度
num_sections: int = 8  # セクション数

# ファイルおよびディレクトリのパス
project_root: str = "/home/connect0459/labo/omu-eddy-covariance"
work_dir: str = f"{project_root}/workspace/footprint"
output_dir: str = f"{work_dir}/private/outputs"  # 出力先のディレクトリ
dotenv_path = f"{work_dir}/.env"  # .envファイル

start_date, end_date = "2024-06-01", "2024-10-31"

if __name__ == "__main__":
    # Monthlyデータの読み込み
    with MonthlyConverter(
        "/home/connect0459/labo/omu-eddy-covariance/workspace/private/monthly",
        file_pattern="SA.Ultra.*.xlsx",
    ) as converter:
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
    all_hotspots: list[HotspotData] = msa.analyze_hotspots(
        duplicate_check_mode="time_window"
    )

    # タイプごとにホットスポットを分類
    gas_hotspots = [spot for spot in all_hotspots if spot.type == "gas"]
    bio_hotspots = [spot for spot in all_hotspots if spot.type == "bio"]

    # インスタンスを作成
    ffa = FluxFootprintAnalyzer(z_m=111)
    df: pd.DataFrame = ffa.combine_all_data(monthly_df, source_type="monthly")

    # # C2H6/CH4比の計算
    # df["Fratio"] = (df["Fc2h6 ultra"] / df["Fch4 ultra"]) / 0.076 * 100

    # CH4フラックスのフットプリント計算
    x_list_ch4, y_list_ch4, c_list_ch4 = ffa.calculate_flux_footprint(
        df=df,
        flux_key="Fch4 ultra",
        plot_count=30000,
    )

    # # C2H6/CH4比のフットプリント計算
    # x_list_ratio, y_list_ratio, c_list_ratio = ffa.calculate_flux_footprint(
    #     df=df,
    #     flux_key="Fratio",
    #     plot_count=10000,
    # )

    fsa = FluxSpatialAnalyzer(8)

    # 解析の実行と結果の保存
    # 1. CH4フラックスとbioホットスポット
    results_ch4_bio = fsa.run_analysis(
        x_list_ch4, y_list_ch4, c_list_ch4, bio_hotspots, num_sections=num_sections
    )
    fig_ch4_bio = fsa.visualize_results(results_ch4_bio)
    fig_ch4_bio.savefig(f"{output_dir}/section_ch4_bio.png", dpi=300)

    # 2. CH4フラックスとgasホットスポット
    results_ch4_gas = fsa.run_analysis(
        x_list_ch4, y_list_ch4, c_list_ch4, gas_hotspots, num_sections=num_sections
    )
    fig_ch4_gas = fsa.visualize_results(results_ch4_gas)
    fig_ch4_gas.savefig(f"{output_dir}/section_ch4_gas.png", dpi=300)

    # # 3. C2H6/CH4比とbioホットスポット
    # results_ratio_bio = fsa.run_analysis(
    #     x_list_ratio,
    #     y_list_ratio,
    #     c_list_ratio,
    #     bio_hotspots,
    #     num_sections=num_sections,
    # )
    # fig_ratio_bio = fsa.visualize_results(results_ratio_bio)
    # fig_ratio_bio.savefig(f"{output_dir}/section_ratio_bio.png", dpi=300)

    # # 4. C2H6/CH4比とgasホットスポット
    # results_ratio_gas = fsa.run_analysis(
    #     x_list_ratio,
    #     y_list_ratio,
    #     c_list_ratio,
    #     gas_hotspots,
    #     num_sections=num_sections,
    # )
    # fig_ratio_gas = fsa.visualize_results(results_ratio_gas)
    # fig_ratio_gas.savefig(f"{output_dir}/section_ratio_gas.png", dpi=300)

    # CH4フラックスの結果を可視化
    fig_ch4 = fsa.visualize_combined_results(
        results_ch4_bio,
        results_ch4_gas,
        title="CH4 Flux Analysis",
        flux_type="CH4 flux",
    )
    fig_ch4.savefig(f"{output_dir}/section_ch4_combined.png", dpi=300)

    # # C2H6/CH4比の結果を可視化
    # fig_ratio = fsa.visualize_combined_results(
    #     results_ratio_bio,
    #     results_ratio_gas,
    #     title="C2H6/CH4 Ratio Analysis",
    #     flux_type="C2H6/CH4 ratio",
    # )
    # fig_ratio.savefig(f"{output_dir}/section_ratio_combined.png", dpi=300)

    # 結果の出力
    print("\nCH4フラックス vs bioホットスポット:")
    for result in results_ch4_bio:
        if result.p_value < 0.05:
            print(
                f"セクター{result.section}: 相関係数 = {result.correlation:.4f}, p値 = {result.p_value:.4f}"
            )

    print("\nCH4フラックス vs gasホットスポット:")
    for result in results_ch4_gas:
        if result.p_value < 0.05:
            print(
                f"セクター{result.section}: 相関係数 = {result.correlation:.4f}, p値 = {result.p_value:.4f}"
            )

    # print("\nC2H6/CH4比 vs bioホットスポット:")
    # for result in results_ratio_bio:
    #     if result.p_value < 0.05:
    #         print(
    #             f"セクター{result.section}: 相関係数 = {result.correlation:.4f}, p値 = {result.p_value:.4f}"
    #         )

    # print("\nC2H6/CH4比 vs gasホットスポット:")
    # for result in results_ratio_gas:
    #     if result.p_value < 0.05:
    #         print(
    #             f"セクター{result.section}: 相関係数 = {result.correlation:.4f}, p値 = {result.p_value:.4f}"
    #         )
