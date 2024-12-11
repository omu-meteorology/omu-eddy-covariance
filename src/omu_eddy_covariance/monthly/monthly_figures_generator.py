import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

    def plot_monthly_c1c2_fluxes_timeseries(
        self,
        monthly_df,
        output_dir: str,
        output_filename: str = "c1c2_fluxes_timeseries.png",
    ):
        """月別のフラックスデータを時系列プロットとして出力する

        Args:
            monthly_df (pd.DataFrame): 月別データを含むDataFrame
            output_dir (str): 出力ファイルを保存するディレクトリのパス
            output_filename (str): 出力ファイルの名前
        """
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
        ax1.scatter(
            monthly_df["Date"], monthly_df["Fch4 ultra"], color="red", alpha=0.5, s=20
        )
        ax1.set_ylabel(r"CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)")
        ax1.set_ylim(-100, 600)
        ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, va="top", fontsize=20)
        ax1.grid(True, alpha=0.3)

        # C2H6フラックスのプロット
        ax2.scatter(
            monthly_df["Date"],
            monthly_df["Fc2h6 ultra"],
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_monthly_psd(
        self,
        input_dir: str,
        output_dir: str,
        fs: float,
        lag_second: float = 0,  # パワースペクトルの計算には関係ないので0とする
        are_inputs_resampled: bool = True,
        file_pattern: str = "*.csv",
        output_filename: str = "monthly_psd.png",
    ) -> None:
        """月間の平均パワースペクトル密度を計算してプロットする

        Args:
            input_dir (str): データファイルが格納されているディレクトリ
            output_dir (str): 出力先ディレクトリ
            file_pattern (str): 処理対象のファイルパターン
        """
        # データの読み込みと結合
        edp = EddyDataPreprocessor()

        # 各変数のパワースペクトルを格納する辞書
        power_spectra = {"Ultra_CH4_ppm_C": [], "Ultra_C2H6_ppb": []}
        freqs = None

        # プログレスバーを表示しながらファイルを処理
        file_list = glob.glob(f"{input_dir}/{file_pattern}")
        for filepath in tqdm(file_list, desc="Processing files"):
            df, _ = edp.get_resampled_df(
                filepath=filepath, is_already_resampled=are_inputs_resampled
            )

            # 各ファイルごとにスペクトル計算
            calculator = SpectrumCalculator(
                df=df,
                fs=fs,
                apply_lag_keys=["Ultra_CH4_ppm_C", "Ultra_C2H6_ppb"],
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
                if freqs is None:
                    freqs = f
                power_spectra[key].append(ps)

        # 各変数のパワースペクトルを平均化
        averaged_spectra = {
            key: np.mean(spectra, axis=0) for key, spectra in power_spectra.items()
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
                "key": "Ultra_CH4_ppm_C",
                "ylabel": r"$fS_{\mathrm{CH_4}} / s_{\mathrm{CH_4}}^2$",
                "color": "red",
                "label": "(a)",
            },
            {
                "key": "Ultra_C2H6_ppb",
                "ylabel": r"$fS_{\mathrm{C_2H_6}} / s_{\mathrm{C_2H_6}}^2$",
                "color": "orange",
                "label": "(b)",
            },
        ]

        # サブプロットの数を2に変更
        _, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        for ax, config in zip(axes, plot_configs):
            # 平均化されたパワースペクトルをプロット
            ax.scatter(
                freqs,
                averaged_spectra[config["key"]],
                c=config["color"],
                s=100,
            )

            # 軸を対数スケールに設定
            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.set_xlim(0.001, 10)

            # -2/3 勾配の参照線
            ax.plot([0.01, 10], [1, 0.01], "-", color="black", alpha=0.5)
            ax.text(0.1, 0.1, "-2/3", fontsize=18)

            ax.set_ylabel(config["ylabel"])
            ax.text(0.02, 0.98, config["label"], transform=ax.transAxes, va="top")
            ax.grid(True, alpha=0.3)

        # x軸ラベルは最後のプロットにのみ表示
        axes[-1].set_xlabel("f (Hz)")

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            f"{output_dir}/{output_filename}",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

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
