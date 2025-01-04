import os
import glob
import jpholiday
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from scipy import linalg
from matplotlib.ticker import FuncFormatter, MultipleLocator
from logging import getLogger, Formatter, Logger, StreamHandler, DEBUG, INFO
from ..ultra.eddydata_preprocessor import EddyDataPreprocessor
from ..ultra.spectrum_calculator import SpectrumCalculator


class MonthlyFiguresGenerator:
    def __init__(
        self,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ) -> None:
        """
        クラスのコンストラクタ

        Args:
            logger (Logger | None): 使用するロガー。Noneの場合は新しいロガーを作成します。
            logging_debug (bool): ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。
        """
        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = MonthlyFiguresGenerator.setup_logger(logger, log_level)

    def plot_c1c2_fluxes_timeseries(
        self,
        df,
        output_dir: str,
        output_filename: str = "timeseries.png",
        datetime_key: str = "Date",
        c1_flux_key: str = "Fch4_ultra",
        c2_flux_key: str = "Fc2h6_ultra",
    ):
        """月別のフラックスデータを時系列プロットとして出力する

        Args:
            df (pd.DataFrame): 月別データを含むDataFrame
            output_dir (str): 出力ファイルを保存するディレクトリのパス
            output_filename (str): 出力ファイルの名前
            datetime_key (str): 日付を含む列の名前。デフォルトは"Date"。
            c1_flux_key (str): CH4フラックスを含む列の名前。デフォルトは"Fch4_ultra"。
            c2_flux_key (str): C2H6フラックスを含む列の名前。デフォルトは"Fc2h6_ultra"。
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # # 図のスタイル設定
        # plt.rcParams.update(
        #     {
        #         "font.size": 20,
        #         "axes.labelsize": 20,
        #         "axes.titlesize": 20,
        #         "xtick.labelsize": 20,
        #         "ytick.labelsize": 20,
        #         "legend.fontsize": 20,
        #     }
        # )

        # 図の作成
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # CH4フラックスのプロット
        ax1.scatter(df[datetime_key], df[c1_flux_key], color="red", alpha=0.5, s=20)
        ax1.set_ylabel(r"CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)")
        ax1.set_ylim(-100, 600)
        ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, va="top", fontsize=20)
        ax1.grid(True, alpha=0.3)

        # C2H6フラックスのプロット
        ax2.scatter(
            df[datetime_key],
            df[c2_flux_key],
            color="orange",
            alpha=0.5,
            s=20,
        )
        ax2.set_ylabel(r"C$_2$H$_6$ flux (nmol m$^{-2}$ s$^{-1}$)")
        ax2.set_ylim(-20, 60)
        ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, va="top", fontsize=20)
        ax2.grid(True, alpha=0.3)

        # x軸の設定
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
        plt.setp(ax2.get_xticklabels(), rotation=0, ha="right")
        ax2.set_xlabel("Month")

        # 図の保存
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_c1c2_concentrations_and_fluxes_timeseries(
        self,
        df: pd.DataFrame,
        output_dir: str,
        output_filename: str = "conc_flux_timeseries.png",
        datetime_key: str = "Date",
        ch4_conc_key: str = "CH4_ultra",
        ch4_flux_key: str = "Fch4_ultra",
        c2h6_conc_key: str = "C2H6_ultra",
        c2h6_flux_key: str = "Fc2h6_ultra",
    ) -> None:
        """
        CH4とC2H6の濃度とフラックスの時系列プロットを作成する

        Args:
            df: 月別データを含むDataFrame
            output_dir: 出力ディレクトリのパス
            output_filename: 出力ファイル名
            datetime_key: 日付列の名前
            ch4_conc_key: CH4濃度列の名前
            ch4_flux_key: CH4フラックス列の名前
            c2h6_conc_key: C2H6濃度列の名前
            c2h6_flux_key: C2H6フラックス列の名前
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # 統計情報の計算と表示
        for name, key in [
            ("CH4 concentration", ch4_conc_key),
            ("CH4 flux", ch4_flux_key),
            ("C2H6 concentration", c2h6_conc_key),
            ("C2H6 flux", c2h6_flux_key),
        ]:
            # NaNを除外してから統計量を計算
            valid_data = df[key].dropna()

            if len(valid_data) > 0:
                percentile_5 = np.nanpercentile(valid_data, 5)
                percentile_95 = np.nanpercentile(valid_data, 95)
                mean_value = np.nanmean(valid_data)
                positive_ratio = (valid_data > 0).mean() * 100

                print(f"\n{name}:")
                print(
                    f"90パーセンタイルレンジ: {percentile_5:.2f} - {percentile_95:.2f}"
                )
                print(f"平均値: {mean_value:.2f}")
                print(f"正の値の割合: {positive_ratio:.1f}%")
            else:
                print(f"\n{name}: データが存在しません")

        # プロットの作成
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

        # CH4濃度のプロット
        ax1.scatter(df[datetime_key], df[ch4_conc_key], color="red", alpha=0.5, s=20)
        ax1.set_ylabel("CH$_4$ Concentration\n(ppm)")
        ax1.set_ylim(1.8, 2.6)
        ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, va="top", fontsize=20)
        ax1.grid(True, alpha=0.3)

        # CH4フラックスのプロット
        ax2.scatter(df[datetime_key], df[ch4_flux_key], color="red", alpha=0.5, s=20)
        ax2.set_ylabel("CH$_4$ Flux\n(nmol m$^{-2}$ s$^{-1}$)")
        ax2.set_ylim(-100, 600)
        # ax2.set_yticks([-100, 0, 200, 400, 600])
        ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, va="top", fontsize=20)
        ax2.grid(True, alpha=0.3)

        # C2H6濃度のプロット
        ax3.scatter(
            df[datetime_key], df[c2h6_conc_key], color="orange", alpha=0.5, s=20
        )
        ax3.set_ylabel("C$_2$H$_6$ Concentration\n(ppb)")
        ax3.text(0.02, 0.98, "(c)", transform=ax3.transAxes, va="top", fontsize=20)
        ax3.grid(True, alpha=0.3)

        # C2H6フラックスのプロット
        ax4.scatter(
            df[datetime_key], df[c2h6_flux_key], color="orange", alpha=0.5, s=20
        )
        ax4.set_ylabel("C$_2$H$_6$ Flux\n(nmol m$^{-2}$ s$^{-1}$)")
        ax4.set_ylim(-20, 40)
        ax4.text(0.02, 0.98, "(d)", transform=ax4.transAxes, va="top", fontsize=20)
        ax4.grid(True, alpha=0.3)

        # x軸の設定
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
        plt.setp(ax4.get_xticklabels(), rotation=0, ha="right")
        ax4.set_xlabel("Month")

        # レイアウトの調整と保存
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_c1c2_fluxes_diurnal_patterns(
        self,
        df: pd.DataFrame,
        y_cols_ch4: list[str],
        y_cols_c2h6: list[str],
        labels_ch4: list[str],
        labels_c2h6: list[str],
        colors_ch4: list[str],
        colors_c2h6: list[str],
        output_dir: str,
        output_filename: str = "diurnal.png",
        legend_only_ch4: bool = False,
        show_label: bool = True,
        show_legend: bool = True,
        subplot_fontsize: int = 20,
        subplot_label_ch4: str | None = "(a)",
        subplot_label_c2h6: str | None = "(b)",
    ) -> None:
        """CH4とC2H6の日変化パターンを1つの図に並べてプロットする"""
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # データの準備
        target_columns = y_cols_ch4 + y_cols_c2h6
        hourly_means, time_points = self._prepare_diurnal_data(df, target_columns)

        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # CH4のプロット (左側)
        ch4_lines = []
        for y_col, label, color in zip(y_cols_ch4, labels_ch4, colors_ch4):
            line = ax1.plot(
                time_points,
                hourly_means["all"][y_col],
                "-o",
                label=label,
                color=color,
            )
            ch4_lines.extend(line)

        # C2H6のプロット (右側)
        c2h6_lines = []
        for y_col, label, color in zip(y_cols_c2h6, labels_c2h6, colors_c2h6):
            line = ax2.plot(
                time_points,
                hourly_means["all"][y_col],
                "o-",
                label=label,
                color=color,
            )
            c2h6_lines.extend(line)

        # 軸の設定
        for ax, ylabel, subplot_label in [
            (ax1, r"CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)", subplot_label_ch4),
            (ax2, r"C$_2$H$_6$ Flux (nmol m$^{-2}$ s$^{-1}$)", subplot_label_c2h6),
        ]:
            self._setup_diurnal_axes(
                ax=ax,
                time_points=time_points,
                ylabel=ylabel,
                subplot_label=subplot_label,
                show_label=show_label,
                show_legend=False,  # 個別の凡例は表示しない
                subplot_fontsize=subplot_fontsize,
            )

        # 軸の追加設定
        ch4_min = min(
            hourly_means["all"][y_cols_ch4].min().min() for y_col in y_cols_ch4
        )
        ch4_max = max(
            hourly_means["all"][y_cols_ch4].max().max() for y_col in y_cols_ch4
        )

        ax1_ylim: tuple[float, float] = min(0, ch4_min - 5), max(100, ch4_max + 5)
        ax1.set_ylim(ax1_ylim)
        ax1.yaxis.set_major_locator(MultipleLocator(20))
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.0f}"))

        # C2H6の軸設定
        c2h6_min = min(
            hourly_means["all"][y_cols_c2h6].min().min() for y_col in y_cols_c2h6
        )
        c2h6_max = max(
            hourly_means["all"][y_cols_c2h6].max().max() for y_col in y_cols_c2h6
        )

        ax2_ylim: tuple[float, float] = min(0, c2h6_min - 0.5), max(3, c2h6_max + 0.5)
        ax2.set_ylim(ax2_ylim)
        ax2.yaxis.set_major_locator(MultipleLocator(1))
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))

        plt.tight_layout()

        # 共通の凡例
        if show_legend:
            all_lines = ch4_lines
            all_labels = [line.get_label() for line in ch4_lines]
            if not legend_only_ch4:
                all_lines += c2h6_lines
                all_labels += [line.get_label() for line in c2h6_lines]
            fig.legend(
                all_lines,
                all_labels,
                loc="center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=len(all_lines),
            )
            plt.subplots_adjust(bottom=0.25)  # 下部に凡例用のスペースを確保

        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_c1c2_fluxes_diurnal_patterns_by_date(
        self,
        df: pd.DataFrame,
        y_col_ch4: str,
        y_col_c2h6: str,
        output_dir: str,
        output_filename: str = "diurnal_by_date.png",
        plot_all: bool = True,
        plot_weekday: bool = True,
        plot_weekend: bool = True,
        plot_holiday: bool = True,
        show_label: bool = True,
        show_legend: bool = True,
        legend_only_ch4: bool = False,
        subplot_fontsize: int = 20,
        subplot_label_ch4: str | None = "(a)",
        subplot_label_c2h6: str | None = "(b)",
    ) -> None:
        """CH4とC2H6の日変化パターンを日付分類して1つの図に並べてプロットする"""
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # データの準備
        target_columns = [y_col_ch4, y_col_c2h6]
        hourly_means, time_points = self._prepare_diurnal_data(
            df, target_columns, include_date_types=True
        )

        # プロットスタイルの設定
        styles = {
            "all": {
                "color": "black",
                "linestyle": "-",
                "alpha": 1.0,
                "label": "All days",
            },
            "weekday": {
                "color": "blue",
                "linestyle": "-",
                "alpha": 0.8,
                "label": "Weekdays",
            },
            "weekend": {
                "color": "red",
                "linestyle": "-",
                "alpha": 0.8,
                "label": "Weekends",
            },
            "holiday": {
                "color": "green",
                "linestyle": "-",
                "alpha": 0.8,
                "label": "Weekends & Holidays",
            },
        }

        # プロット対象の条件を選択
        plot_conditions = {
            "all": plot_all,
            "weekday": plot_weekday,
            "weekend": plot_weekend,
            "holiday": plot_holiday,
        }
        selected_conditions = {
            key: means
            for key, means in hourly_means.items()
            if key in plot_conditions and plot_conditions[key]
        }

        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # CH4とC2H6のプロット用のラインオブジェクトを保存
        ch4_lines = []
        c2h6_lines = []

        # CH4とC2H6のプロット
        for condition, means in selected_conditions.items():
            style = styles[condition].copy()  # スタイルをコピーして変更

            # CH4プロット（実線）
            line_ch4 = ax1.plot(time_points, means[y_col_ch4], marker="o", **style)
            ch4_lines.extend(line_ch4)

            # C2H6プロット（破線）
            style["linestyle"] = "--"  # C2H6用に破線に変更
            line_c2h6 = ax2.plot(time_points, means[y_col_c2h6], marker="o", **style)
            c2h6_lines.extend(line_c2h6)

        # 軸の設定
        for ax, ylabel, subplot_label in [
            (ax1, r"CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)", subplot_label_ch4),
            (ax2, r"C$_2$H$_6$ Flux (nmol m$^{-2}$ s$^{-1}$)", subplot_label_c2h6),
        ]:
            self._setup_diurnal_axes(
                ax=ax,
                time_points=time_points,
                ylabel=ylabel,
                subplot_label=subplot_label,
                show_label=show_label,
                show_legend=False,
                subplot_fontsize=subplot_fontsize,
            )

        # 軸の追加設定
        ch4_min = min(means[y_col_ch4].min() for means in selected_conditions.values())
        ch4_max = max(means[y_col_ch4].max() for means in selected_conditions.values())

        ax1_ylim: tuple[float, float] = min(0, ch4_min - 5), max(100, ch4_max + 5)
        ax1.set_ylim(ax1_ylim)
        ax1.yaxis.set_major_locator(MultipleLocator(20))
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.0f}"))

        # C2H6の軸設定
        c2h6_min = min(
            means[y_col_c2h6].min() for means in selected_conditions.values()
        )
        c2h6_max = max(
            means[y_col_c2h6].max() for means in selected_conditions.values()
        )

        ax2_ylim: tuple[float, float] = min(0, c2h6_min - 0.5), max(3, c2h6_max + 0.5)
        ax2.set_ylim(ax2_ylim)
        ax2.yaxis.set_major_locator(MultipleLocator(1))
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))

        plt.tight_layout()

        # 共通の凡例を図の下部に配置
        if show_legend:
            lines_to_show = (
                ch4_lines if legend_only_ch4 else ch4_lines[: len(selected_conditions)]
            )
            fig.legend(
                lines_to_show,
                [
                    style["label"]
                    for style in list(styles.values())[: len(lines_to_show)]
                ],
                loc="center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=len(lines_to_show),
            )
            plt.subplots_adjust(bottom=0.25)  # 下部に凡例用のスペースを確保

        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_flux_diurnal_patterns_with_std(
        self,
        df: pd.DataFrame,
        output_dir: str,
        ch4_flux_key: str = "Fch4",
        c2h6_flux_key: str = "Fc2h6",
        ch4_label: str = r"$\mathregular{CH_{4}}$フラックス",
        c2h6_label: str = r"$\mathregular{C_{2}H_{6}}$フラックス",
        datetime_key: str = "Date",
        output_filename: str = "diurnal_patterns.png",
        window_size: int = 6,  # 移動平均の窓サイズ
        show_std: bool = True,  # 標準偏差の表示有無
        alpha_std: float = 0.1,  # 標準偏差の透明度
    ) -> None:
        """CH4とC2H6フラックスの日変化パターンをプロットする

        Args:
            df (pd.DataFrame): データフレーム
            output_dir (str): 出力ディレクトリのパス
            ch4_flux_key (str): CH4フラックスのカラム名
            c2h6_flux_key (str): C2H6フラックスのカラム名
            ch4_label (str): CH4フラックスのラベル
            c2h6_label (str): C2H6フラックスのラベル
            datetime_key (str): 日時カラムの名前
            output_filename (str): 出力ファイル名
            window_size (int): 移動平均の窓サイズ（デフォルト6）
            show_std (bool): 標準偏差を表示するかどうか
            alpha_std (float): 標準偏差の透明度（0-1）
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # # プロットのスタイル設定
        # plt.rcParams.update({
        #     'font.size': 20,
        #     'axes.labelsize': 20,
        #     'axes.titlesize': 20,
        #     'xtick.labelsize': 20,
        #     'ytick.labelsize': 20,
        #     'legend.fontsize': 20,
        # })

        # 日時インデックスの処理
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df[datetime_key] = pd.to_datetime(df[datetime_key])
            df.set_index(datetime_key, inplace=True)

        # 時刻データの抽出とグループ化
        df["hour"] = df.index.hour
        hourly_means = df.groupby("hour")[[ch4_flux_key, c2h6_flux_key]].agg(
            ["mean", "std"]
        )

        # 24時間目のデータ点を追加（0時のデータを使用）
        last_hour = hourly_means.iloc[0:1].copy()
        last_hour.index = [24]
        hourly_means = pd.concat([hourly_means, last_hour])

        # 24時間分のデータポイントを作成
        time_points = pd.date_range("2024-01-01", periods=25, freq="H")

        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 移動平均の計算と描画
        ch4_mean = (
            hourly_means[(ch4_flux_key, "mean")]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )
        c2h6_mean = (
            hourly_means[(c2h6_flux_key, "mean")]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )

        if show_std:
            ch4_std = (
                hourly_means[(ch4_flux_key, "std")]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )
            c2h6_std = (
                hourly_means[(c2h6_flux_key, "std")]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )

            ax1.fill_between(
                time_points,
                ch4_mean - ch4_std,
                ch4_mean + ch4_std,
                color="blue",
                alpha=alpha_std,
            )
            ax2.fill_between(
                time_points,
                c2h6_mean - c2h6_std,
                c2h6_mean + c2h6_std,
                color="red",
                alpha=alpha_std,
            )

        # メインのラインプロット
        ax1.plot(time_points, ch4_mean, "blue", label=ch4_label)
        ax2.plot(time_points, c2h6_mean, "red", label=c2h6_label)

        # 軸の設定
        for ax, ylabel in [
            (ax1, r"CH$_4$ (nmol m$^{-2}$ s$^{-1}$)"),
            (ax2, r"C$_2$H$_6$ (nmol m$^{-2}$ s$^{-1}$)"),
        ]:
            ax.set_xlabel("Time")
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%-H"))
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
            ax.set_xlim(time_points[0], time_points[-1])
            ax.grid(True, alpha=0.3)
            ax.legend()

        # グラフの保存
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        # 統計情報の表示（オプション）
        for key, name in [(ch4_flux_key, "CH4"), (c2h6_flux_key, "C2H6")]:
            mean_val = hourly_means[(key, "mean")].mean()
            min_val = hourly_means[(key, "mean")].min()
            max_val = hourly_means[(key, "mean")].max()
            min_time = hourly_means[(key, "mean")].idxmin()
            max_time = hourly_means[(key, "mean")].idxmax()

            self.logger.info(f"{name} Statistics:")
            self.logger.info(f"Mean: {mean_val:.2f}")
            self.logger.info(f"Min: {min_val:.2f} (Hour: {min_time})")
            self.logger.info(f"Max: {max_val:.2f} (Hour: {max_time})")
            self.logger.info(f"Max/Min ratio: {max_val/min_val:.2f}\n")

    def plot_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        output_dir: str,
        output_filename: str = "scatter.png",
        xlabel: str | None = None,
        ylabel: str | None = None,
        show_label: bool = True,
        x_axis_range: tuple | None = None,
        y_axis_range: tuple | None = None,
        fixed_slope: float = 0.076,
        show_fixed_slope: bool = False,
    ) -> None:
        """散布図を作成し、TLS回帰直線を描画します。

        引数:
            df (pd.DataFrame): プロットに使用するデータフレーム
            x_col (str): x軸に使用する列名
            y_col (str): y軸に使用する列名
            xlabel (str): x軸のラベル
            ylabel (str): y軸のラベル
            output_dir (str): 出力先ディレクトリ
            output_filename (str, optional): 出力ファイル名。デフォルトは"scatter.png"
            show_label (bool, optional): 軸ラベルを表示するかどうか。デフォルトはTrue
            x_axis_range (tuple, optional): x軸の範囲。デフォルトはNone。
            y_axis_range (tuple, optional): y軸の範囲。デフォルトはNone。
            fixed_slope (float, optional): 固定傾きを指定するための値。デフォルトは0.076
            show_fixed_slope (bool, optional): 固定傾きの線を表示するかどうか。デフォルトはFalse
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # 有効なデータの抽出
        df = MonthlyFiguresGenerator.get_valid_data(df, x_col, y_col)

        # データの準備
        x = df[x_col].values
        y = df[y_col].values

        # データの中心化
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_c = x - x_mean
        y_c = y - y_mean

        # TLS回帰の計算
        data_matrix = np.vstack((x_c, y_c))
        cov_matrix = np.cov(data_matrix)
        _, eigenvecs = linalg.eigh(cov_matrix)
        largest_eigenvec = eigenvecs[:, -1]

        slope = largest_eigenvec[1] / largest_eigenvec[0]
        intercept = y_mean - slope * x_mean

        # R²とRMSEの計算
        y_pred = slope * x + intercept
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        # プロットの作成
        fig, ax = plt.subplots(figsize=(6, 6))

        # データ点のプロット
        ax.scatter(x, y, color="black")

        # データの範囲を取得
        if x_axis_range is None:
            x_axis_range = (df[x_col].min(), df[x_col].max())
        if y_axis_range is None:
            y_axis_range = (df[y_col].min(), df[y_col].max())

        # 回帰直線のプロット
        x_range = np.linspace(x_axis_range[0], x_axis_range[1], 150)
        y_range = slope * x_range + intercept
        ax.plot(x_range, y_range, "r", label="TLS regression")

        # 傾き固定の線を追加（フラグがTrueの場合）
        if show_fixed_slope:
            fixed_intercept = (
                y_mean - fixed_slope * x_mean
            )  # 中心点を通るように切片を計算
            y_fixed = fixed_slope * x_range + fixed_intercept
            ax.plot(x_range, y_fixed, "b--", label=f"Slope = {fixed_slope}", alpha=0.7)

        # 軸の設定
        ax.set_xlim(x_axis_range)
        ax.set_ylim(y_axis_range)

        if show_label:
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)

        # 1:1の関係を示す点線（軸の範囲が同じ場合のみ表示）
        if (
            x_axis_range is not None
            and y_axis_range is not None
            and x_axis_range == y_axis_range
        ):
            ax.plot(
                [x_axis_range[0], x_axis_range[1]],
                [x_axis_range[0], x_axis_range[1]],
                "k--",
                alpha=0.5,
            )

        # 回帰情報の表示
        equation = (
            f"y = {slope:.2f}x {'+' if intercept >= 0 else '-'} {abs(intercept):.2f}"
        )
        position_x = 0.05
        fig_ha: str = "left"
        ax.text(
            position_x,
            0.95,
            equation,
            transform=ax.transAxes,
            va="top",
            ha=fig_ha,
            color="red",
        )
        ax.text(
            position_x,
            0.88,
            f"R² = {r_squared:.2f}",
            transform=ax.transAxes,
            va="top",
            ha=fig_ha,
            color="red",
        )
        ax.text(
            position_x,
            0.81,  # RMSEのための新しい位置
            f"RMSE = {rmse:.2f}",
            transform=ax.transAxes,
            va="top",
            ha=fig_ha,
            color="red",
        )
        # 目盛り線の設定
        ax.grid(True, alpha=0.3)

        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_source_contributions_diurnal(
        self,
        df: pd.DataFrame,
        output_dir: str,
        ch4_flux_key: str,
        c2h6_flux_key: str,
        gas_label: str = "都市ガス起源",
        bio_label: str = "生物起源",
        datetime_key: str = "Date",
        output_filename: str = "source_contributions.png",
        window_size: int = 6,  # 移動平均の窓サイズ
        show_stats: bool = True,  # 統計情報を表示するかどうか,
        smooth: bool = False,
        y_max: float = 100,  # y軸の上限値を追加
    ) -> None:
        """CH4フラックスの都市ガス起源と生物起源の日変化を積み上げグラフとして表示

        Args:
            df (pd.DataFrame): データフレーム
            output_dir (str): 出力ディレクトリのパス
            ch4_flux_key (str): CH4フラックスのカラム名
            c2h6_flux_key (str): C2H6フラックスのカラム名
            gas_label (str): 都市ガス起源のラベル
            bio_label (str): 生物起源のラベル
            datetime_key (str): 日時カラムの名前
            output_filename (str): 出力ファイル名
            window_size (int): 移動平均の窓サイズ
            show_stats (bool): 統計情報を表示するかどうか
            smooth (bool): 移動平均を適用するかどうか
            y_max (float): y軸の上限値（デフォルト: 100）
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # 起源の計算
        df_with_sources = self._calculate_source_contributions(
            df=df,
            ch4_flux_key=ch4_flux_key,
            c2h6_flux_key=c2h6_flux_key,
            datetime_key=datetime_key,
        )

        # 時刻データの抽出とグループ化
        df_with_sources["hour"] = df_with_sources.index.hour
        hourly_means = df_with_sources.groupby("hour")[["ch4_gas", "ch4_bio"]].mean()

        # 24時間目のデータ点を追加（0時のデータを使用）
        last_hour = hourly_means.iloc[0:1].copy()
        last_hour.index = [24]
        hourly_means = pd.concat([hourly_means, last_hour])

        # 移動平均の適用
        hourly_means_smoothed = hourly_means
        if smooth:
            hourly_means_smoothed = hourly_means.rolling(
                window=window_size, center=True, min_periods=1
            ).mean()

        # 24時間分のデータポイントを作成
        time_points = pd.date_range("2024-01-01", periods=25, freq="H")

        # プロットの作成
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # 積み上げプロット
        ax.fill_between(
            time_points,
            0,
            hourly_means_smoothed["ch4_bio"],
            color="green",
            alpha=0.5,
            label=bio_label,
        )
        ax.fill_between(
            time_points,
            hourly_means_smoothed["ch4_bio"],
            hourly_means_smoothed["ch4_bio"] + hourly_means_smoothed["ch4_gas"],
            color="orange",
            alpha=0.5,
            label=gas_label,
        )

        # 合計値のライン
        total_flux = hourly_means_smoothed["ch4_bio"] + hourly_means_smoothed["ch4_gas"]
        ax.plot(time_points, total_flux, "-", color="black", alpha=0.5)

        # 軸の設定
        ax.set_xlabel("Time")
        ax.set_ylabel(r"CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-H"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
        ax.set_xlim(time_points[0], time_points[-1])
        ax.set_ylim(0, y_max)  # y軸の範囲を設定
        ax.grid(True, alpha=0.3)
        ax.legend()

        # グラフの保存
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        # 統計情報の表示
        if show_stats:
            stats = {
                "都市ガス起源": hourly_means["ch4_gas"],
                "生物起源": hourly_means["ch4_bio"],
                "合計": hourly_means["ch4_gas"] + hourly_means["ch4_bio"],
            }

            for source, data in stats.items():
                mean_val = data.mean()
                min_val = data.min()
                max_val = data.max()
                min_time = data.idxmin()
                max_time = data.idxmax()

                self.logger.info(f"{source}の統計:")
                print(f"  平均値: {mean_val:.2f}")
                print(f"  最小値: {min_val:.2f} (Hour: {min_time})")
                print(f"  最大値: {max_val:.2f} (Hour: {max_time})")
                if min_val != 0:
                    print(f"  最大/最小比: {max_val/min_val:.2f}")

    def plot_spectra(
        self,
        input_dir: str,
        output_dir: str,
        fs: float,
        lag_second: float,
        ch4_key: str = "Ultra_CH4_ppm_C",
        c2h6_key: str = "Ultra_C2H6_ppb",
        label_ch4: str | None = None,
        label_c2h6: str | None = None,
        are_inputs_resampled: bool = True,
        file_pattern: str = "*.csv",
        output_basename: str = "spectrum",
        plot_power: bool = True,
        plot_co: bool = True,
        markersize: float = 14,
    ) -> None:
        """月間の平均パワースペクトル密度を計算してプロットする。

        データファイルを指定されたディレクトリから読み込み、パワースペクトル密度を計算し、
        結果を指定された出力ディレクトリにプロットして保存します。

        Args:
            input_dir (str): データファイルが格納されているディレクトリ。
            output_dir (str): 出力先ディレクトリ。
            fs (float): サンプリング周波数。
            lag_second (float, optional): ラグ時間（秒）。
            ch4_key (str, optional): CH4の濃度データが入ったカラムのキー。
            c2h6_key (str, optional):C2H6の濃度データが入ったカラムのキー。
            are_inputs_resampled (bool, optional): 入力データが再サンプリングされているかどうか。デフォルトはTrue。
            file_pattern (str, optional): 処理対象のファイルパターン。デフォルトは"*.csv"。
            output_filename (str, optional): 出力ファイル名。デフォルトは"psd.png"。
        """
        # データの読み込みと結合
        edp = EddyDataPreprocessor()

        # 各変数のパワースペクトルを格納する辞書
        power_spectra = {ch4_key: [], c2h6_key: []}
        co_spectra = {ch4_key: [], c2h6_key: []}
        freqs = None

        # プログレスバーを表示しながらファイルを処理
        file_list = glob.glob(os.path.join(input_dir, file_pattern))
        for filepath in tqdm(file_list, desc="Processing files"):
            df, _ = edp.get_resampled_df(
                filepath=filepath, is_already_resampled=are_inputs_resampled
            )

            # 風速成分の計算を追加
            df = edp.add_uvw_columns(df)

            # NaNや無限大を含む行を削除
            df = df.replace([np.inf, -np.inf], np.nan).dropna(
                subset=[ch4_key, c2h6_key, "wind_w"]
            )

            # データが十分な行数を持っているか確認
            if len(df) < 100:
                continue

            # 各ファイルごとにスペクトル計算
            calculator = SpectrumCalculator(
                df=df,
                fs=fs,
                apply_lag_keys=[ch4_key, c2h6_key],
                lag_second=lag_second,
            )

            # 各変数のパワースペクトルを計算して保存
            for key in power_spectra.keys():
                f, ps = calculator.calculate_power_spectrum(
                    key=key,
                    dimensionless=True,
                    frequency_weighted=True,
                    interpolate_points=True,
                    scaling="density",
                )
                # 最初のファイル処理時にfreqsを初期化
                if freqs is None:
                    freqs = f
                    power_spectra[key].append(ps)
                # 以降は周波数配列の長さが一致する場合のみ追加
                elif len(f) == len(freqs):
                    power_spectra[key].append(ps)

                # コスペクトル
                _, cs, _ = calculator.calculate_co_spectrum(
                    key1="wind_w",
                    key2=key,
                    dimensionless=True,
                    frequency_weighted=True,
                    interpolate_points=True,
                    # scaling="density",
                    scaling="spectrum",
                )
                if freqs is not None and len(cs) == len(freqs):
                    co_spectra[key].append(cs)

        # 各変数のスペクトルを平均化
        averaged_power_spectra = {
            key: np.mean(spectra, axis=0) for key, spectra in power_spectra.items()
        }
        averaged_co_spectra = {
            key: np.mean(spectra, axis=0) for key, spectra in co_spectra.items()
        }

        # # プロット設定
        # plt.rcParams.update(
        #     {
        #         "font.size": 20,
        #         "axes.labelsize": 20,
        #         "axes.titlesize": 20,
        #         "xtick.labelsize": 20,
        #         "ytick.labelsize": 20,
        #         "legend.fontsize": 20,
        #     }
        # )

        # プロット設定を修正
        plot_configs = [
            {
                "key": ch4_key,
                "psd_ylabel": r"$fS_{\mathrm{CH_4}} / s_{\mathrm{CH_4}}^2$",
                "co_ylabel": r"$fCo_{w\mathrm{CH_4}} / (\sigma_w \sigma_{\mathrm{CH_4}})$",
                "color": "red",
                "label": label_ch4,
            },
            {
                "key": c2h6_key,
                "psd_ylabel": r"$fS_{\mathrm{C_2H_6}} / s_{\mathrm{C_2H_6}}^2$",
                "co_ylabel": r"$fCo_{w\mathrm{C_2H_6}} / (\sigma_w \sigma_{\mathrm{C_2H_6}})$",
                "color": "orange",
                "label": label_c2h6,
            },
        ]

        # # パワースペクトルの図を作成
        # if plot_power:
        #     _, axes_psd = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        #     for ax, config in zip(axes_psd, plot_configs):
        #         ax.scatter(
        #             freqs,
        #             averaged_power_spectra[config["key"]],
        #             c=config["color"],
        #             s=100,
        #         )
        #         ax.set_xscale("log")
        #         ax.set_yscale("log")
        #         ax.set_xlim(0.001, 10)
        #         ax.plot([0.01, 10], [1, 0.01], "-", color="black", alpha=0.5)
        #         ax.text(0.1, 0.06, "-2/3", fontsize=18)
        #         ax.set_ylabel(config["psd_ylabel"])
        #         if config["label"] is not None:
        #             ax.text(
        #                 0.02, 0.98, config["label"], transform=ax.transAxes, va="top"
        #             )
        #         ax.grid(True, alpha=0.3)
        #         ax.set_xlabel("f (Hz)")

        # パワースペクトルの図を作成
        if plot_power:
            _, axes_psd = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
            for ax, config in zip(axes_psd, plot_configs):
                ax.plot(
                    freqs,
                    averaged_power_spectra[config["key"]],
                    "o",  # マーカーを丸に設定
                    color=config["color"],
                    markersize=markersize,
                )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(0.001, 10)
                ax.plot([0.01, 10], [1, 0.01], "-", color="black", alpha=0.5)
                ax.text(0.1, 0.06, "-2/3", fontsize=18)
                ax.set_ylabel(config["psd_ylabel"])
                if config["label"] is not None:
                    ax.text(
                        0.02, 0.98, config["label"], transform=ax.transAxes, va="top"
                    )
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("f (Hz)")

            plt.tight_layout()
            os.makedirs(output_dir, exist_ok=True)
            output_path_psd: str = os.path.join(
                output_dir, f"power_{output_basename}.png"
            )
            plt.savefig(
                output_path_psd,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # # コスペクトルの図を作成
        # if plot_co:
        #     _, axes_cosp = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        #     for ax, config in zip(axes_cosp, plot_configs):
        #         ax.scatter(
        #             freqs,
        #             averaged_co_spectra[config["key"]],
        #             c=config["color"],
        #             s=100,
        #         )
        #         ax.set_xscale("log")
        #         ax.set_yscale("log")
        #         ax.set_xlim(0.001, 10)
        #         ax.plot([0.01, 10], [1, 0.01], "-", color="black", alpha=0.5)
        #         ax.text(0.1, 0.1, "-4/3", fontsize=18)
        #         ax.set_ylabel(config["co_ylabel"])
        #         if config["label"] is not None:
        #             ax.text(
        #                 0.02, 0.98, config["label"], transform=ax.transAxes, va="top"
        #             )
        #         ax.grid(True, alpha=0.3)
        #         ax.set_xlabel("f (Hz)")

        # コスペクトルの図を作成
        if plot_co:
            _, axes_cosp = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
            for ax, config in zip(axes_cosp, plot_configs):
                ax.plot(
                    freqs,
                    averaged_co_spectra[config["key"]],
                    "o",  # マーカーを丸に設定
                    color=config["color"],
                    markersize=markersize,
                )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(0.001, 10)
                ax.plot([0.01, 10], [1, 0.01], "-", color="black", alpha=0.5)
                ax.text(0.1, 0.1, "-4/3", fontsize=18)
                ax.set_ylabel(config["co_ylabel"])
                if config["label"] is not None:
                    ax.text(
                        0.02, 0.98, config["label"], transform=ax.transAxes, va="top"
                    )
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("f (Hz)")

            plt.tight_layout()
            output_path_csd: str = os.path.join(output_dir, f"co_{output_basename}.png")
            plt.savefig(
                output_path_csd,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def plot_turbulence(
        self,
        df: pd.DataFrame,
        output_dir: str,
        output_filename: str = "turbulence.png",
        uz_key: str = "Uz",
        ch4_key: str = "Ultra_CH4_ppm_C",
        c2h6_key: str = "Ultra_C2H6_ppb",
        timestamp_key: str = "TIMESTAMP",
        add_serial_labels: bool = True,
    ) -> None:
        """時系列データのプロットを作成する

        Args:
            df (pd.DataFrame): プロットするデータを含むDataFrame
            output_dir (str): 出力ディレクトリのパス
            output_filename (str): 出力ファイル名
            uz_key (str): 鉛直風速データのカラム名
            ch4_key (str): メタンデータのカラム名
            c2h6_key (str): エタンデータのカラム名
            timestamp_key (str): タイムスタンプのカラム名
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # データの前処理
        df = df.copy()

        # タイムスタンプをインデックスに設定（まだ設定されていない場合）
        if not isinstance(df.index, pd.DatetimeIndex):
            df[timestamp_key] = pd.to_datetime(df[timestamp_key])
            df.set_index(timestamp_key, inplace=True)

        # 開始時刻と終了時刻を取得
        start_time = df.index[0]
        end_time = df.index[-1]

        # 開始時刻の分を取得
        start_minute = start_time.minute

        # 時間軸の作成（実際の開始時刻からの経過分数）
        minutes_elapsed = (df.index - start_time).total_seconds() / 60

        # プロットの作成
        _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # 鉛直風速
        ax1.plot(minutes_elapsed, df[uz_key], "k-", linewidth=0.5)
        ax1.set_ylabel(r"$w$ (m s$^{-1}$)")
        if add_serial_labels:
            ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, va="top")
        ax1.grid(True, alpha=0.3)

        # CH4濃度
        ax2.plot(minutes_elapsed, df[ch4_key], "r-", linewidth=0.5)
        ax2.set_ylabel(r"$\mathrm{CH_4}$ (ppm)")
        if add_serial_labels:
            ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, va="top")
        ax2.grid(True, alpha=0.3)

        # C2H6濃度
        ax3.plot(minutes_elapsed, df[c2h6_key], "orange", linewidth=0.5)
        ax3.set_ylabel(r"$\mathrm{C_2H_6}$ (ppb)")
        if add_serial_labels:
            ax3.text(0.02, 0.98, "(c)", transform=ax3.transAxes, va="top")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel("Time (minutes)")

        # x軸の範囲を実際の開始時刻から30分後までに設定
        total_minutes = (end_time - start_time).total_seconds() / 60
        ax3.set_xlim(0, min(30, total_minutes))

        # x軸の目盛りを5分間隔で設定
        np.arange(start_minute, start_minute + 35, 5)
        ax3.xaxis.set_major_locator(MultipleLocator(5))

        # レイアウトの調整
        plt.tight_layout()

        # 図の保存
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_wind_rose_sources(
        self,
        df: pd.DataFrame,
        output_dir: str,
        ch4_flux_key: str = "Fch4",
        c2h6_flux_key: str = "Fc2h6",
        wind_dir_key: str = "Wind direction",
        gas_label: str = "都市ガス起源",
        bio_label: str = "生物起源",
        datetime_key: str = "Date",
        output_filename: str = "wind_rose.png",
        num_directions: int = 8,  # 方位の数（8方位）
        subplot_label: str = "(a)",
        show_stats: bool = True,  # 統計情報を表示するかどうか
    ) -> None:
        """CH4フラックスの都市ガス起源と生物起源の風配図を作成

        Args:
            df (pd.DataFrame): データフレーム
            output_dir (str): 出力ディレクトリのパス
            ch4_flux_key (str): CH4フラックスのカラム名
            c2h6_flux_key (str): C2H6フラックスのカラム名
            wind_dir_key (str): 風向のカラム名
            gas_label (str): 都市ガス起源のラベル
            bio_label (str): 生物起源のラベル
            datetime_key (str): 日時カラムの名前
            output_filename (str): 出力ファイル名
            num_directions (int): 方位の数（デフォルト8）
            subplot_label (str): サブプロットのラベル
            show_stats (bool): 統計情報を表示するかどうか
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # 起源の計算
        df_with_sources = self._calculate_source_contributions(
            df=df,
            ch4_flux_key=ch4_flux_key,
            c2h6_flux_key=c2h6_flux_key,
            datetime_key=datetime_key,
        )

        # 方位の定義
        direction_ranges = self._define_direction_ranges(num_directions)

        # 方位ごとのデータを集計
        direction_data = self._aggregate_direction_data(
            df_with_sources, wind_dir_key, direction_ranges
        )

        # プロットの作成
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="polar")

        # 方位の角度（ラジアン）を計算
        theta = np.array(
            [np.radians(angle) for angle in direction_data["center_angle"]]
        )

        # 生物起源のプロット
        ax.bar(
            theta,
            direction_data["bio_flux"],
            width=np.radians(360 / num_directions),
            bottom=0.0,
            color="green",
            alpha=0.5,
            label=bio_label,
        )

        # 都市ガス起源のプロット（積み上げ）
        ax.bar(
            theta,
            direction_data["gas_flux"],
            width=np.radians(360 / num_directions),
            bottom=direction_data["bio_flux"],
            color="orange",
            alpha=0.5,
            label=gas_label,
        )

        # 方位ラベルの設定
        ax.set_theta_zero_location("N")  # 北を上に設定
        ax.set_theta_direction(-1)  # 時計回りに設定

        # 方位ラベルの表示
        labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        angles = np.radians(np.linspace(0, 360, len(labels), endpoint=False))
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)

        # サブプロットラベルの追加（デフォルトは左上）
        if subplot_label:
            ax.text(0.05, 1.05, subplot_label, transform=ax.transAxes, fontsize=16)

        # 凡例の表示
        ax.legend(loc="center left", bbox_to_anchor=(1.2, 0.5))

        # 単位の追加
        ax.text(
            0.95,
            0.05,
            r"[nmol m$^{-2}$ s$^{-1}$]",
            transform=ax.transAxes,
            fontsize=14,
            ha="right",
        )

        # グラフの保存
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        # 統計情報の表示
        if show_stats:
            for source in ["gas", "bio"]:
                flux_data = direction_data[f"{source}_flux"]
                mean_val = flux_data.mean()
                max_val = flux_data.max()
                max_dir = direction_data.loc[flux_data.idxmax(), "name"]

                self.logger.info(
                    f'{gas_label if source == "gas" else bio_label}の統計:'
                )
                print(f"  平均フラックス: {mean_val:.2f}")
                print(f"  最大フラックス: {max_val:.2f}")
                print(f"  最大フラックスの方位: {max_dir}")

    def _define_direction_ranges(self, num_directions: int = 8) -> pd.DataFrame:
        """方位の範囲を定義

        Args:
            num_directions (int): 方位の数（デフォルト8）

        Returns:
            pd.DataFrame: 方位の定義を含むDataFrame
        """
        # 8方位の場合の方位名と中心角度
        if num_directions == 8:
            directions = pd.DataFrame(
                {
                    "name": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                    "center_angle": [0, 45, 90, 135, 180, 225, 270, 315],
                }
            )
        else:
            raise ValueError(f"現在{num_directions}方位はサポートされていません")

        # 各方位の範囲を計算
        angle_range = 360 / num_directions
        directions["start_angle"] = directions["center_angle"] - angle_range / 2
        directions["end_angle"] = directions["center_angle"] + angle_range / 2

        # -180度から180度の範囲に正規化
        directions["start_angle"] = np.where(
            directions["start_angle"] > 180,
            directions["start_angle"] - 360,
            directions["start_angle"],
        )
        directions["end_angle"] = np.where(
            directions["end_angle"] > 180,
            directions["end_angle"] - 360,
            directions["end_angle"],
        )

        return directions

    def _aggregate_direction_data(
        self,
        df: pd.DataFrame,
        wind_dir_key: str,
        direction_ranges: pd.DataFrame,
    ) -> pd.DataFrame:
        """方位ごとのフラックスデータを集計

        Args:
            df (pd.DataFrame): ソース分離済みのデータフレーム
            wind_dir_key (str): 風向のカラム名
            direction_ranges (pd.DataFrame): 方位の定義

        Returns:
            pd.DataFrame: 方位ごとの集計データ
        """
        result_data = direction_ranges.copy()
        result_data["gas_flux"] = 0.0
        result_data["bio_flux"] = 0.0

        for idx, row in direction_ranges.iterrows():
            if row["start_angle"] < row["end_angle"]:
                mask = (df[wind_dir_key] > row["start_angle"]) & (
                    df[wind_dir_key] <= row["end_angle"]
                )
            else:  # 北方向など、-180度と180度をまたぐ場合
                mask = (df[wind_dir_key] > row["start_angle"]) | (
                    df[wind_dir_key] <= row["end_angle"]
                )

            result_data.loc[idx, "gas_flux"] = df.loc[mask, "ch4_gas"].mean()
            result_data.loc[idx, "bio_flux"] = df.loc[mask, "ch4_bio"].mean()

        # NaNを0に置換
        result_data = result_data.fillna(0)

        return result_data

    def _calculate_source_contributions(
        self,
        df: pd.DataFrame,
        ch4_flux_key: str,
        c2h6_flux_key: str,
        gas_ratio_c1c2: float = 0.076,
        datetime_key: str = "Date",
    ) -> pd.DataFrame:
        """CH4フラックスの都市ガス起源と生物起源の寄与を計算する。
        このロジックでは、燃焼起源のCH4フラックスは考慮せず計算している。

        Args:
            df (pd.DataFrame): 入力データフレーム
            ch4_flux_key (str): CH4フラックスのカラム名
            c2h6_flux_key (str): C2H6フラックスのカラム名
            gas_ratio_c1c2 (float): ガスのC2H6/CH4比（ppb/ppb）
            datetime_key (str): 日時カラムの名前

        Returns:
            pd.DataFrame: 起源別のフラックス値を含むデータフレーム
        """
        df = df.copy()

        # 日時インデックスの処理
        if not isinstance(df.index, pd.DatetimeIndex):
            df[datetime_key] = pd.to_datetime(df[datetime_key])
            df.set_index(datetime_key, inplace=True)

        # C2H6/CH4比の計算
        df["c2h6_ch4_ratio"] = df[c2h6_flux_key] / df[ch4_flux_key]

        # 都市ガスの標準組成に基づく都市ガス比率の計算
        df["gas_ratio"] = df["c2h6_ch4_ratio"] / gas_ratio_c1c2 * 100

        # gas_ratioに基づいて都市ガス起源と生物起源の寄与を比例配分
        df["ch4_gas"] = df[ch4_flux_key] * np.clip(df["gas_ratio"] / 100, 0, 1)
        df["ch4_bio"] = df[ch4_flux_key] * (1 - np.clip(df["gas_ratio"] / 100, 0, 1))

        return df

    def _prepare_diurnal_data(
        self,
        df: pd.DataFrame,
        target_columns: list[str],
        include_date_types: bool = False,
    ) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
        """日変化パターンの計算に必要なデータを準備する

        Args:
            df (pd.DataFrame): 入力データフレーム
            target_columns (list[str]): 計算対象の列名のリスト
            include_date_types (bool): 日付タイプ（平日/休日など）の分類を含めるかどうか

        Returns:
            tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
                - 時間帯ごとの平均値を含むDataFrameの辞書
                - 24時間分の時間点
        """
        df = df.copy()
        df["hour"] = pd.to_datetime(df["Date"]).dt.hour

        # 時間ごとの平均値を計算する関数
        def calculate_hourly_means(data_df, condition=None):
            if condition is not None:
                data_df = data_df[condition]
            return data_df.groupby("hour")[target_columns].mean().reset_index()

        # 基本の全日データを計算
        hourly_means = {"all": calculate_hourly_means(df)}

        # 日付タイプによる分類が必要な場合
        if include_date_types:
            dates = pd.to_datetime(df["Date"])
            is_weekend = dates.dt.dayofweek.isin([5, 6])
            is_holiday = dates.map(lambda x: jpholiday.is_holiday(x.date()))
            is_weekday = ~(is_weekend | is_holiday)

            hourly_means.update(
                {
                    "weekday": calculate_hourly_means(df, is_weekday),
                    "weekend": calculate_hourly_means(df, is_weekend),
                    "holiday": calculate_hourly_means(df, is_weekend | is_holiday),
                }
            )

        # 24時目のデータを追加
        for key in hourly_means:
            last_row = hourly_means[key].iloc[0:1].copy()
            last_row["hour"] = 24
            hourly_means[key] = pd.concat(
                [hourly_means[key], last_row], ignore_index=True
            )

        # 24時間分のデータポイントを作成
        time_points = pd.date_range("2024-01-01", periods=25, freq="h")

        return hourly_means, time_points

    def _setup_diurnal_axes(
        self,
        ax: plt.Axes,
        time_points: pd.DatetimeIndex,
        ylabel: str,
        subplot_label: str | None = None,
        show_label: bool = True,
        show_legend: bool = True,
        subplot_fontsize: int = 20,
    ) -> None:
        """日変化プロットの軸の設定を行う

        Args:
            ax (plt.Axes): 設定対象の軸
            time_points (pd.DatetimeIndex): 時間軸のポイント
            ylabel (str): y軸のラベル
            subplot_label (str | None): サブプロットのラベル
            show_label (bool): 軸ラベルを表示するかどうか
            show_legend (bool): 凡例を表示するかどうか
            subplot_fontsize (int): サブプロットのフォントサイズ
        """
        if show_label:
            ax.set_xlabel("Time")
            ax.set_ylabel(ylabel)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-H"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
        ax.set_xlim(time_points[0], time_points[-1])
        ax.set_xticks(time_points[::6])
        ax.set_xticklabels(["0", "6", "12", "18", "24"])

        if subplot_label:
            ax.text(
                0.02,
                0.98,
                subplot_label,
                transform=ax.transAxes,
                va="top",
                fontsize=subplot_fontsize,
            )

        if show_legend:
            ax.legend()

    @staticmethod
    def get_valid_data(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
        """
        指定された列の有効なデータ（NaNを除いた）を取得します。

        引数:
            df (DataFrame): データフレーム
            x_col (str): X軸の列名
            y_col (str): Y軸の列名

        戻り値:
            pd.DataFrame: 有効なデータのみを含むDataFrame
        """
        return df.copy().dropna(subset=[x_col, y_col])

    @staticmethod
    def setup_logger(logger: Logger | None, log_level: int = INFO) -> Logger:
        """
        ロガーを設定します。

        このメソッドは、ロギングの設定を行い、ログメッセージのフォーマットを指定します。
        ログメッセージには、日付、ログレベル、メッセージが含まれます。

        渡されたロガーがNoneまたは不正な場合は、新たにロガーを作成し、標準出力に
        ログメッセージが表示されるようにStreamHandlerを追加します。ロガーのレベルは
        引数で指定されたlog_levelに基づいて設定されます。

        引数:
            logger (Logger | None): 使用するロガー。Noneの場合は新しいロガーを作成します。
            log_level (int): ロガーのログレベル。デフォルトはINFO。

        戻り値:
            Logger: 設定されたロガーオブジェクト。
        """
        if logger is not None and isinstance(logger, Logger):
            return logger
        # 渡されたロガーがNoneまたは正しいものでない場合は独自に設定
        new_logger: Logger = getLogger()
        # 既存のハンドラーをすべて削除
        for handler in new_logger.handlers[:]:
            new_logger.removeHandler(handler)
        new_logger.setLevel(log_level)  # ロガーのレベルを設定
        ch = StreamHandler()
        ch_formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(ch_formatter)  # フォーマッターをハンドラーに設定
        new_logger.addHandler(ch)  # StreamHandlerの追加
        return new_logger

    @staticmethod
    def setup_plot_params(
        font_family: list[str] = ["Arial", "Dejavu Sans"],
        font_size: float = 20,
        legend_size: float = 20,
        tick_size: float = 20,
        title_size: float = 20,
        plot_params=None,
    ) -> None:
        """
        matplotlibのプロットパラメータを設定します。

        引数:
            font_family (list[str]): 使用するフォントファミリーのリスト。
            font_size (float): 軸ラベルのフォントサイズ。
            legend_size (float): 凡例のフォントサイズ。
            tick_size (float): 軸目盛りのフォントサイズ。
            title_size (float): タイトルのフォントサイズ。
            plot_params (Optional[Dict[str, any]]): matplotlibのプロットパラメータの辞書。
        """
        # デフォルトのプロットパラメータ
        default_params = {
            "axes.linewidth": 1.0,
            "axes.titlesize": title_size,  # タイトル
            "grid.color": "gray",
            "grid.linewidth": 1.0,
            "font.family": font_family,
            "font.size": font_size,  # 軸ラベル
            "legend.fontsize": legend_size,  # 凡例
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "xtick.labelsize": tick_size,  # 軸目盛
            "ytick.labelsize": tick_size,  # 軸目盛
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "ytick.direction": "out",
            "ytick.major.width": 1.0,
        }

        # plot_paramsが定義されている場合、デフォルトに追記
        if plot_params:
            default_params.update(plot_params)

        plt.rcParams.update(default_params)  # プロットパラメータを更新
