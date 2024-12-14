import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from scipy import stats
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
        """
        CH4とC2H6の日変化パターンを1つの図に並べてプロットする。
        各時間の値は、その月の同じ時間帯のデータの平均値として計算される。

        例: 0時の値は、その月の0:00-0:59のすべてのデータの平均値

        Args:
            df (pd.DataFrame): 日変化パターンをプロットするためのデータフレーム。
            y_cols_ch4 (list[str]): CH4フラックスの列名リスト。
            y_cols_c2h6 (list[str]): C2H6フラックスの列名リスト。
            labels_ch4 (list[str]): CH4フラックスのラベルリスト。
            labels_c2h6 (list[str]): C2H6フラックスのラベルリスト。
            colors_ch4 (list[str]): CH4フラックスの色リスト。
            colors_c2h6 (list[str]): C2H6フラックスの色リスト。
            output_dir (str): 出力先ディレクトリ。
            output_filename (str, optional): 出力ファイル名。デフォルトは"diurnal.png"。
            legend_only_ch4 (bool, optional): CH4のみの凡例を表示するかどうか。デフォルトはFalse。
            show_label (bool, optional): 軸ラベルを表示するかどうか。デフォルトはTrue。
            show_legend (bool, optional): 凡例を表示するかどうか。デフォルトはTrue。
            subplot_fontsize (int, optional): サブプロットのフォントサイズ。デフォルトは20。
            subplot_label_ch4 (str | None, optional): CH4のサブプロットラベル。デフォルトは"(a)"。
            subplot_label_c2h6 (str | None, optional): C2H6のサブプロットラベル。デフォルトは"(b)"。

        Returns:
            None
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        df = df.copy()

        # 時間データの抽出と平均値の計算
        self.logger.debug("Calculating hourly means")
        df["hour"] = pd.to_datetime(df["Date"]).dt.hour

        # 時間ごとの平均値を計算
        hourly_means = (
            df.groupby("hour")
            .agg(
                {
                    **{col: "mean" for col in y_cols_ch4},
                    **{col: "mean" for col in y_cols_c2h6},
                }
            )
            .reset_index()
        )

        # 24時目のデータを0時のデータで補完
        last_row = hourly_means.iloc[0:1].copy()
        last_row["hour"] = 24
        hourly_means = pd.concat([hourly_means, last_row], ignore_index=True)

        # 24時間分のデータを作成（0-23時）
        time_points = pd.date_range("2024-01-01", periods=25, freq="h")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # CH4のプロット (左側)
        for y_col, label, color in zip(y_cols_ch4, labels_ch4, colors_ch4):
            ax1.plot(time_points, hourly_means[y_col], "-o", label=label, color=color)

        if show_label:
            ax1.set_xlabel("Time")
            ax1.set_ylabel(r"CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)")

        # CH4のプロット (左側)の軸設定
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%-H"))  # %-Hで先頭の0を削除
        ax1.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
        ax1.set_xlim(time_points[0], time_points[-1])
        # 24時の表示を修正
        ax1.set_xticks(time_points[::6])
        ax1.set_xticklabels(["0", "6", "12", "18", "24"])

        # CH4のy軸の設定
        ch4_max = hourly_means[y_cols_ch4].max().max()
        if ch4_max < 100:
            ax1.set_ylim(0, 100)

        ax1.yaxis.set_major_locator(MultipleLocator(20))
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.0f}"))
        if subplot_label_ch4 is not None:
            ax1.text(
                0.02,
                0.98,
                subplot_label_ch4,
                transform=ax1.transAxes,
                va="top",
                fontsize=subplot_fontsize,
            )

        # C2H6のプロット (右側)
        for y_col, label, color in zip(y_cols_c2h6, labels_c2h6, colors_c2h6):
            ax2.plot(time_points, hourly_means[y_col], "-o", label=label, color=color)

        if show_label:
            ax2.set_xlabel("Time")
            ax2.set_ylabel(r"C$_2$H$_6$ Flux (nmol m$^{-2}$ s$^{-1}$)")

        # CH4のプロット (左側)
        ch4_lines = []  # 凡例用にラインオブジェクトを保存
        for y_col, label, color in zip(y_cols_ch4, labels_ch4, colors_ch4):
            (line,) = ax1.plot(
                time_points, hourly_means[y_col], "-o", label=label, color=color
            )
            ch4_lines.append(line)

        # C2H6のプロット (右側)
        c2h6_lines = []  # 凡例用にラインオブジェクトを保存
        for y_col, label, color in zip(y_cols_c2h6, labels_c2h6, colors_c2h6):
            (line,) = ax2.plot(
                time_points, hourly_means[y_col], "-o", label=label, color=color
            )
            c2h6_lines.append(line)

        # 個別の凡例を削除し、図の下部に共通の凡例を配置
        if show_legend:
            all_lines = ch4_lines
            all_labels = labels_ch4
            if not legend_only_ch4:
                all_lines += c2h6_lines
                all_labels += labels_c2h6
            fig.legend(
                all_lines,
                all_labels,
                loc="center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=len(all_lines),
            )

        # C2H6のプロット (右側)の軸設定
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%-H"))  # %-Hで先頭の0を削除
        ax2.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
        ax2.set_xlim(time_points[0], time_points[-1])
        # 24時の表示を修正
        ax2.set_xticks(time_points[::6])
        ax2.set_xticklabels(["0", "6", "12", "18", "24"])

        ax2.yaxis.set_major_locator(MultipleLocator(1))
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))
        if subplot_label_c2h6 is not None:
            ax2.text(
                0.02,
                0.98,
                subplot_label_c2h6,
                transform=ax2.transAxes,
                va="top",
                fontsize=subplot_fontsize,
            )

        plt.tight_layout()
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
        """散布図と回帰直線をプロットする

        Args:
            df (pd.DataFrame): プロットに使用するデータフレーム。
            x_col (str): x軸に使用する列名。
            y_col (str): y軸に使用する列名。
            xlabel (str): x軸のラベル。
            ylabel (str): y軸のラベル。
            output_dir (str): 出力先ディレクトリ。
            output_filename (str, optional): 出力ファイル名。デフォルトは"scatter.png"。
            show_label (bool, optional): 軸ラベルを表示するかどうか。デフォルトはTrue。
            axis_range (tuple, optional): x軸とy軸の範囲。デフォルトは(-50, 200)。
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        df = MonthlyFiguresGenerator.get_valid_data(df, x_col, y_col)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(df[x_col], df[y_col], color="black")

        # 線形回帰
        slope, intercept, r_value, _, _ = stats.linregress(df[x_col], df[y_col])

        # 近似直線
        x_range = np.linspace(axis_range[0], axis_range[1], 150)
        y_range = slope * x_range + intercept
        ax.plot(x_range, y_range, "r")

        if show_label:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        ax.set_xlim(axis_range)
        ax.set_ylim(axis_range)

        # 1:1の関係を示す点線を追加
        ax.plot(
            [axis_range[0], axis_range[1]],
            [axis_range[0], axis_range[1]],
            "k--",
            alpha=0.5,
        )

        # 近似直線の式と決定係数を表示
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
            f"R² = {r_value**2:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            color="red",
        )

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
