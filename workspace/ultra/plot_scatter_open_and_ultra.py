import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from matplotlib.ticker import FuncFormatter, MultipleLocator
from logging import getLogger, Logger, StreamHandler, Formatter, INFO, DEBUG


class FluxPlotter:
    def __init__(
        self,
        output_dir: str,
        labelsize: float = 20,
        ticksize: float = 20,
        plot_params: dict | None = None,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        散布図と日変化パターンを作図するクラスを初期化します。

        引数:
            output_dir (str): 図を保存するディレクトリのパス
            labelsize (float): 軸ラベルのフォントサイズ。デフォルトは20。
            ticksize (float): 軸目盛りのフォントサイズ。デフォルトは16。
            plot_params (dict | None): matplotlibのプロットパラメータを指定する辞書。
            logger (Optional[Logger]): 使用するロガー。Noneの場合は新しいロガーを生成。
            logging_debug (bool): ログレベルをDEBUGに設定するかどうか。デフォルトはFalse。
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 図表の初期設定
        FluxPlotter.setup_plot_params(labelsize, ticksize, plot_params)

        # ロガーの設定
        log_level = DEBUG if logging_debug else INFO
        self.logger = FluxPlotter.setup_logger(logger, log_level)

        # DataFrameの初期化
        self.__df: pd.DataFrame | None = None

    @property
    def df(self) -> pd.DataFrame | None:
        """処理済みのDataFrameを取得"""
        return self.__df

    def load_data(self, file_path: str) -> None:
        """
        CSVファイルを読み込み、必要な前処理を行います。

        引数:
            file_path (str): 読み込むCSVファイルのパス
        """
        self.logger.debug(f"Loading data from: {file_path}")

        # データの読み込みと前処理

        df = pd.read_csv(file_path, skiprows=[1])
        df["Time"] = pd.to_datetime(df["Date"])

        # 数値データを含む列を数値型に変換
        numeric_columns = [
            "Fch4_open",
            "Fch4_ultra",
            "Fc2h6_ultra",
            "Fch4_picaro",
        ]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

        self.__df = df

        self.logger.info(f"Data loaded and processed successfully: {len(df)} rows")

    def get_valid_data(self, x_col: str, y_col: str) -> pd.DataFrame:
        """
        指定された列の有効なデータ（NaNを除いた）を取得します。

        引数:
            x_col (str): X軸の列名
            y_col (str): Y軸の列名

        戻り値:
            pd.DataFrame: 有効なデータのみを含むDataFrame
        """
        if self.__df is None:
            raise ValueError(
                "No data loaded. Please load data first using load_data()."
            )

        return self.__df.copy().dropna(subset=[x_col, y_col])

    def plot_combined_diurnal_patterns(
        self,
        y_cols_ch4: list[str],
        y_cols_c2h6: list[str],
        labels_ch4: list[str],
        labels_c2h6: list[str],
        colors_ch4: list[str],
        colors_c2h6: list[str],
        filename: str,
        label_only_ch4: bool = False,
        show_label: bool = True,
        show_legend: bool = True,
        subplot_fontsize: int = 20,
        subplot_label_ch4: str = "(a)",
        subplot_label_c2h6: str = "(b)",
    ) -> None:
        """
        CH4とC2H6の日変化パターンを1つの図に並べてプロットする。
        各時間の値は、その月の同じ時間帯のデータの平均値として計算される。

        例：0時の値は、その月の0:00-0:59のすべてのデータの平均値
        """
        df = self.__df.copy()
        if df is None:
            raise ValueError("No data loaded. Please load data first using load_data()")

        self.logger.debug("Creating combined diurnal patterns plot")

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
            if not label_only_ch4:
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
        self.save_figure(fig, filename)

    def plot_scatter_with_regression(
        self,
        x_col: str,
        y_col: str,
        xlabel: str,
        ylabel: str,
        filename: str,
        show_label: bool = True,
        axis_range: tuple = (-50, 200),
    ) -> None:
        """散布図と回帰直線をプロットする"""
        if self.__df is None:
            raise ValueError(
                "No data loaded. Please load data first using load_data()."
            )

        self.logger.debug("Creating scatter plot with regression")

        numeric_df = self.get_valid_data(x_col, y_col)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(numeric_df[x_col], numeric_df[y_col], color="black")

        # 線形回帰
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            numeric_df[x_col], numeric_df[y_col]
        )

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

        self.save_figure(fig, filename)

    def save_figure(self, fig: plt.Figure, filename: str) -> None:
        """図を指定されたディレクトリに保存する"""
        output_path = os.path.join(self.output_dir, filename)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Figure saved: {output_path}")

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
        labelsize: float, ticksize: float, plot_params: dict | None = None
    ) -> None:
        """matplotlibのプロットパラメータを設定する"""
        default_params = {
            "font.family": "Arial",
            "font.size": labelsize,
            "xtick.labelsize": ticksize,
            "ytick.labelsize": ticksize,
        }

        if plot_params:
            default_params.update(plot_params)

        plt.rcParams.update(default_params)


if __name__ == "__main__":
    months: list[str] = ["06", "07", "08", "09", "10"]
    subplot_labels: list[list[str]] = [
        ["(a)", "(b)"],
        ["(c)", "(d)"],
        ["(e)", "(f)"],
        ["(g)", "(h)"],
        ["(i)", "(j)"],
    ]

    # プロッターの初期化
    plotter = FluxPlotter(
        output_dir="/home/connect0459/labo/omu-eddy-covariance/workspace/ultra/private/outputs/diurnals",
        labelsize=24,
        ticksize=24,
        logging_debug=False,
    )

    for i, month in enumerate(months):  # インデックスを取得してループ
        # データの読み込み
        plotter.load_data(
            f"/home/connect0459/labo/omu-eddy-covariance/workspace/ultra/private/data/diurnals/diurnals-{month}.csv"
        )

        # 日変化パターン
        plotter.plot_combined_diurnal_patterns(
            y_cols_ch4=["Fch4_ultra", "Fch4_open", "Fch4_picaro"],
            y_cols_c2h6=["Fc2h6_ultra"],
            labels_ch4=["Ultra", "Open Path", "G2401"],
            labels_c2h6=["Ultra"],
            label_only_ch4=True,
            show_label=False,
            show_legend=(month == "09" or month == "07"),
            colors_ch4=["black", "red", "blue"],
            colors_c2h6=["black"],
            filename=f"diurnal-pattern-slide-{month}.png",
            subplot_label_ch4=subplot_labels[i][0],
            subplot_label_c2h6=subplot_labels[i][1],
            subplot_fontsize=24,
        )

        # 散布図
        plotter.plot_scatter_with_regression(
            x_col="Fch4_open",
            y_col="Fch4_ultra",
            xlabel=r"Open Path CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
            ylabel=r"Ultra CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
            filename=f"scatter-open_ultra-{month}.png",
        )

        plotter.plot_scatter_with_regression(
            x_col="Fch4_open",
            y_col="Fch4_picaro",
            xlabel=r"Open Path CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
            ylabel=r"Picarro CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
            filename=f"scatter-open_picarro-{month}.png",
        )

        plotter.plot_scatter_with_regression(
            x_col="Fch4_picaro",
            y_col="Fch4_ultra",
            xlabel=r"Picarro CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
            ylabel=r"Ultra CH$_4$ Flux (nmol m$^{-2}$ s$^{-1}$)",
            filename=f"scatter-picarro_ultra-{month}.png",
        )
