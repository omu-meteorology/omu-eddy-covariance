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

        # 図のスタイル設定
        plt.rcParams.update(
            {
                "font.family": ["Arial"],
                "font.size": 20,
                "axes.labelsize": 20,
                "axes.titlesize": 20,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
                "legend.fontsize": 20,
            }
        )

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
                "-o",
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

        # プロットスタイルの設定（すべて実線に変更）
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
            style = styles[condition]
            # CH4プロット
            line_ch4 = ax1.plot(time_points, means[y_col_ch4], marker="o", **style)
            ch4_lines.extend(line_ch4)

            # C2H6プロット
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
        c2h6_min = min(means[y_col_c2h6].min() for means in selected_conditions.values())
        c2h6_max = max(means[y_col_c2h6].max() for means in selected_conditions.values())

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

    def plot_c1c2_fluxes_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        xlabel: str,
        ylabel: str,
        output_dir: str,
        output_filename: str = "scatter.png",
        show_label: bool = True,
        axis_range: tuple = (-50, 200),
    ) -> None:
        """散布図とTLS回帰直線をプロットする

        Args:
            df (pd.DataFrame): プロットに使用するデータフレーム
            x_col (str): x軸に使用する列名
            y_col (str): y軸に使用する列名
            xlabel (str): x軸のラベル
            ylabel (str): y軸のラベル
            output_dir (str): 出力先ディレクトリ
            output_filename (str, optional): 出力ファイル名。デフォルトは"scatter.png"
            show_label (bool, optional): 軸ラベルを表示するかどうか。デフォルトはTrue
            axis_range (tuple, optional): x軸とy軸の範囲。デフォルトは(-50, 200)
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

        # R²の計算
        y_pred = slope * x + intercept
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

        # プロットの作成
        fig, ax = plt.subplots(figsize=(6, 6))

        # データ点のプロット
        ax.scatter(x, y, color="black")

        # 回帰直線のプロット
        x_range = np.linspace(axis_range[0], axis_range[1], 150)
        y_range = slope * x_range + intercept
        ax.plot(x_range, y_range, "r", label="TLS regression")

        if show_label:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        # 軸の設定
        ax.set_xlim(axis_range)
        ax.set_ylim(axis_range)

        # 1:1の関係を示す点線
        ax.plot(
            [axis_range[0], axis_range[1]],
            [axis_range[0], axis_range[1]],
            "k--",
            alpha=0.5,
        )

        # 回帰情報の表示
        equation = (
            f"y = {slope:.2f}x {'+' if intercept >= 0 else '-'} {abs(intercept):.2f}"
        )
        position_x = 0.50
        ax.text(
            position_x,
            0.95,
            equation,
            transform=ax.transAxes,
            va="top",
            ha="right",
            color="red",
        )
        ax.text(
            position_x,
            0.88,
            f"R² = {r_squared:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            color="red",
        )

        # 目盛り線の設定
        ax.grid(True, alpha=0.3)

        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_spectra(
        self,
        input_dir: str,
        output_dir: str,
        fs: float,
        lag_second: float,
        ch4_key: str = "Ultra_CH4_ppm_C",
        c2h6_key: str = "Ultra_C2H6_ppb",
        are_inputs_resampled: bool = True,
        file_pattern: str = "*.csv",
        output_basename: str = "spectrum",
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

        # プロット設定
        plt.rcParams.update(
            {
                "font.family": ["Arial"],
                "font.size": 20,
                "axes.labelsize": 20,
                "axes.titlesize": 20,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
                "legend.fontsize": 20,
            }
        )

        # プロット設定を修正（Tvを削除）
        plot_configs = [
            {
                "key": ch4_key,
                "psd_ylabel": r"$fS_{\mathrm{CH_4}} / s_{\mathrm{CH_4}}^2$",
                "co_ylabel": r"$fCo_{w\mathrm{CH_4}} / (\sigma_w \sigma_{\mathrm{CH_4}})$",
                "color": "red",
                "label": "(a)",
            },
            {
                "key": c2h6_key,
                "psd_ylabel": r"$fS_{\mathrm{C_2H_6}} / s_{\mathrm{C_2H_6}}^2$",
                "co_ylabel": r"$fCo_{w\mathrm{C_2H_6}} / (\sigma_w \sigma_{\mathrm{C_2H_6}})$",
                "color": "orange",
                "label": "(b)",
            },
        ]

        # パワースペクトルの図を作成
        _, axes_psd = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        for ax, config in zip(axes_psd, plot_configs):
            ax.scatter(
                freqs,
                averaged_power_spectra[config["key"]],
                c=config["color"],
                s=100,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.001, 10)
            ax.plot([0.01, 10], [1, 0.01], "-", color="black", alpha=0.5)
            ax.text(0.1, 0.1, "-2/3", fontsize=18)
            ax.set_ylabel(config["psd_ylabel"])
            ax.text(0.02, 0.98, config["label"], transform=ax.transAxes, va="top")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("f (Hz)")

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        output_path_psd: str = os.path.join(output_dir, f"power_{output_basename}.png")
        plt.savefig(
            output_path_psd,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # コスペクトルの図を作成
        _, axes_cosp = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        for ax, config in zip(axes_cosp, plot_configs):
            ax.scatter(
                freqs,
                averaged_co_spectra[config["key"]],
                c=config["color"],
                s=100,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.001, 10)
            ax.plot([0.01, 10], [1, 0.01], "-", color="black", alpha=0.5)
            ax.text(0.1, 0.1, "-4/3", fontsize=18)
            ax.set_ylabel(config["co_ylabel"])
            ax.text(0.02, 0.98, config["label"], transform=ax.transAxes, va="top")
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
