import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class TransferFunctionCalculator:
    """
    このクラスは、CSVファイルからデータを読み込み、処理し、
    伝達関数を計算してプロットするための機能を提供します。

    この実装は Moore (1986) の論文に基づいています。
    """

    def __init__(
        self,
        file_path: str,
        freq_key: str,
        cutoff_freq_low: float,
        cutoff_freq_high: float,
    ):
        """
        TransferFunctionCalculatorクラスのコンストラクタ。

        Args:
            file_path (str): 分析対象のCSVファイルのパス。
            freq_key (str): 周波数のキー。
            cutoff_freq_low (float): カットオフ周波数の最低値
            cutoff_freq_high (float): カットオフ周波数の最高値
        """
        self.freq_key: str = freq_key
        self.cutoff_freq_low: float = cutoff_freq_low
        self.cutoff_freq_high: float = cutoff_freq_high
        self.df: pd.DataFrame = self.__load_data(file_path)
        self.setup_plot_params()

    @classmethod
    def transfer_function(cls, x: np.ndarray, a: float) -> np.ndarray:
        """
        伝達関数を計算する。

        Args:
            x (np.ndarray): 周波数の配列。
            a (float): 伝達関数の係数。

        Returns:
            np.ndarray: 伝達関数の値。
        """
        return np.exp(-np.log(np.sqrt(2)) * np.power(x / a, 2))

    def __load_data(self, file_path: str) -> pd.DataFrame:
        """
        CSVファイルからデータを読み込む。

        Args:
            filepath (str): csvファイルのパス。

        Returns:
            pd.DataFrame: 読み込まれたデータフレーム。
        """
        tmp = pd.read_csv(file_path, header=None, nrows=1, skiprows=0)
        header = tmp.loc[tmp.index[0]]
        df = pd.read_csv(file_path, header=None, skiprows=1)
        df.columns = header
        return df

    def __cutoff_df(self, df) -> pd.DataFrame:
        """
        カットオフ周波数に基づいてDataFrameを加工するメソッド
        """
        df_cutoff: pd.DataFrame = df.loc[
            (self.cutoff_freq_low <= df.index) & (df.index <= self.cutoff_freq_high)
        ]
        return df_cutoff

    def setup_plot_params(self) -> None:
        """
        Matplotlibのプロットパラメータを設定する。

        日本語フォントの設定やグラフの外観設定を行う。
        """
        plt.rcParams["font.sans-serif"] = ["MS Gothic"] + plt.rcParams[
            "font.sans-serif"
        ]
        plt.rcParams.update(
            {
                "axes.edgecolor": "black",
                "axes.labelcolor": "black",
                "text.color": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "grid.color": "gray",
                "axes.grid": False,
                "ytick.major.size": 0,
                "ytick.direction": "out",
                "ytick.major.width": 0,
                "axes.linewidth": 1.5,
                "grid.linewidth": 1.0,
                "font.size": 13,
                "axes.labelsize": 18,
            }
        )

    def process_data(self, reference_key: str, target_key: str) -> pd.DataFrame:
        """
        指定されたキーに基づいてデータを処理する。

        Args:
            reference_key (str): 参照データのカラム名。
            target_key (str): ターゲットデータのカラム名。

        Returns:
            pd.DataFrame: 処理されたデータフレーム。
        """
        freq_key: str = self.freq_key

        # データ型の確認と変換
        self.df[freq_key] = pd.to_numeric(self.df[freq_key], errors="coerce")
        self.df[reference_key] = pd.to_numeric(self.df[reference_key], errors="coerce")
        self.df[target_key] = pd.to_numeric(self.df[target_key], errors="coerce")

        # NaNを含む行を削除
        self.df = self.df.dropna(subset=[freq_key, reference_key, target_key])

        # グループ化と中央値の計算
        grouped = self.df.groupby(freq_key)
        reference_data = grouped[reference_key].median()
        target_data = grouped[target_key].median()

        df_processed = pd.DataFrame(
            {"reference": reference_data, "target": target_data}
        )

        # 異常な比率を除去
        df_processed.loc[
            (
                (df_processed["target"] / df_processed["reference"] > 1)
                | (df_processed["target"] / df_processed["reference"] < 0)
            )
        ] = np.nan
        df_processed = df_processed.dropna()

        return df_processed

    def calculate_transfer_function(self, df_processed: pd.DataFrame) -> float:
        """
        伝達関数の係数を計算する。

        Args:
            df_processed (pd.DataFrame): 処理されたデータフレーム。
            f_low (float, optional): 下限周波数。デフォルトは0.001。
            f_high (float, optional): 上限周波数。デフォルトは10。

        Returns:
            float: 伝達関数の係数a。
        """
        df_cutoff: pd.DataFrame = self.__cutoff_df(df_processed)

        array_x = np.array(df_cutoff.index)
        array_y = np.array(df_cutoff["target"] / df_cutoff["reference"])

        param, _ = curve_fit(self.transfer_function, array_x, array_y)
        return param[0]

    def plot_ratio(
        self, df_processed: pd.DataFrame, target_name: str, reference_name: str
    ) -> None:
        """
        ターゲットと参照の比率をプロットする。

        Args:
            df_processed (pd.DataFrame): 処理されたデータフレーム。
            target_name (str): ターゲットの名前。
            reference_name (str): 参照の名前。
        """
        plt.figure(figsize=(10, 6))
        plt.plot(
            df_processed.index, df_processed["target"] / df_processed["reference"], "o"
        )
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.xlabel("f (Hz)")
        plt.ylabel(f"{target_name} / {reference_name}")
        plt.title(f"{target_name}と{reference_name}の比")
        plt.show()

    def plot_cospectra(
        self,
        key1: str,
        key2: str,
        label1: str = "データ1",
        label2: str = "データ2",
        color1: str = "gray",
        color2: str = "red",
        subplot_label: str | None = "(a)",
    ) -> None:
        """
        2種類のコスペクトルをプロットする。

        Args:
            key1 (str): 1つ目のコスペクトルデータのカラム名。
            key2 (str): 2つ目のコスペクトルデータのカラム名。
            label1 (str, optional): 1つ目のデータのラベル名。デフォルトは"データ1"。
            label2 (str, optional): 2つ目のデータのラベル名。デフォルトは"データ2"。
            color1 (str, optional): 1つ目のデータの色。デフォルトは'gray'。
            color2 (str, optional): 2つ目のデータの色。デフォルトは'red'。
            subplot_label (str | None, optional): 左上に表示するサブプロットラベル。デフォルトは"(a)"。
        """
        data1 = self.df[self.df[key1] > 0].groupby(self.freq_key)[key1].median()
        data2 = self.df[self.df[key2] > 0].groupby(self.freq_key)[key2].median()

        plt.figure(figsize=(10, 6))

        # データ1のプロット
        plt.plot(data1.index, data1, "o", color=color1, label=label1)

        # データ2のプロット
        plt.plot(data2.index, data2, "o", color=color2, label=label2)

        # -4/3 勾配の参照線
        plt.plot([0.01, 10], [10, 0.001], "-", color="black")

        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.0001, 10)
        ax.set_ylim(0.0001, 10)
        ax.set_xlabel("f (Hz)", size=16)
        ax.set_ylabel("無次元コスペクトル", size=16)
        ax.grid(color="gray", linestyle="--")

        ax.legend(
            bbox_to_anchor=(0.05, 1),
            loc="lower left",
            fontsize=13,
            ncol=3,
            frameon=False,
        )
        if subplot_label is not None:
            ax.text(0.00015, 3, subplot_label, fontsize=18)
        ax.text(0.25, 0.4, "-4/3", fontsize=18)

        plt.tight_layout()
        plt.show()

    def plot_transfer_function(
        self,
        df_processed: pd.DataFrame,
        a: float,
        target_name: str,
        reference_name: str,
    ) -> None:
        """
        伝達関数とそのフィットをプロットする。

        Args:
            df_processed (pd.DataFrame): 処理されたデータフレーム。
            a (float): 伝達関数の係数。
            target_name (str): ターゲットの名前。
            reference_name (str): 参照の名前。
            f_low (float, optional): 下限周波数。デフォルトは0.001。
            f_high (float, optional): 上限周波数。デフォルトは10。
        """
        df_cutoff: pd.DataFrame = self.__cutoff_df(df_processed)

        plt.figure(figsize=(10, 6))
        plt.plot(
            df_cutoff.index,
            df_cutoff["target"] / df_cutoff["reference"],
            "o",
            label=f"{target_name} / {reference_name}",
        )

        x_fit = np.logspace(
            np.log10(self.cutoff_freq_low), np.log10(self.cutoff_freq_high), 1000
        )
        y_fit = self.transfer_function(x_fit, a)
        plt.plot(x_fit, y_fit, "-", label=f"フィット (a = {a:.4f})")

        ax = plt.gca()
        ax.set_xscale("log")
        plt.xlabel("f (Hz)")
        plt.ylabel("コスペクトル比")
        plt.legend()
        plt.show()

    def analyze_transfer_function(
        self,
        reference_key: str,
        reference_name: str,
        target_key: str,
        target_name: str,
        show_tf_plot: bool = True,
    ) -> float:
        """
        指定されたターゲットの伝達関数を分析する。

        この方法は、データの処理、比率のプロット、伝達関数の計算、
        およびフィットのプロットを含む完全な分析プロセスを実行する。

        Args:
            reference_key (str): 参照データのカラム名。
            reference_name (str): 参照の名前（例：'Tv'）。
            target_key (str): ターゲットデータのカラム名。
            target_name (str): ターゲットの名前（例：'CH4', 'C2H6'）。
            show_tf_plot (bool): 伝達関数のグラフの表示オプション。

        Return:
            a (float): 伝達関数の係数'a'。
        """
        df_processed = self.process_data(reference_key, target_key)
        # self.plot_ratio(df_processed, target_name, reference_name)
        a: float = self.calculate_transfer_function(df_processed)
        if show_tf_plot:
            self.plot_transfer_function(df_processed, a, target_name, reference_name)
        return a
