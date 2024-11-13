import jpholiday
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime
from logging import getLogger, Formatter, Logger, StreamHandler, DEBUG, INFO
from PIL import Image
from tqdm import tqdm


class FluxFootprintAnalyzer:
    """
    フラックスフットプリントを解析および可視化するクラス。

    このクラスは、フラックスデータの処理、フットプリントの計算、
    および結果を衛星画像上に可視化するメソッドを提供します。
    座標系と単位に関する重要な注意:
    - すべての距離はメートル単位で計算されます
    - 座標系の原点(0,0)は測定タワーの位置に対応します
    - x軸は東西方向（正が東）
    - y軸は南北方向（正が北）
    - 風向は気象学的風向（北から時計回りに測定）を使用

    この実装は、Kormann and Meixner (2001)の論文に基づいています。
    """

    def __init__(
        self,
        z_m: float,
        labelsize: float = 20,
        ticksize: float = 16,
        plot_params=None,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        衛星画像を用いて FluxFootprintAnalyzer を初期化します。

        引数:
            z_m (float): 測定の高さ（メートル単位）。
            labelsize (float): 軸ラベルのフォントサイズ。デフォルトは20。
            ticksize (float): 軸目盛りのフォントサイズ。デフォルトは16。
            plot_params (Optional[Dict[str, any]]): matplotlibのプロットパラメータを指定する辞書。
            logger (Logger | None): 使用するロガー。Noneの場合は新しいロガーを生成します。
            logging_debug (bool): ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。
        """
        # 必要なカラムのキー名
        self.required_columns: list[str] = [
            "Date",
            "WS vector",
            "u*",
            "z/L",
            "Wind direction",
            "sigmaV",
        ]
        # 定数や共通の変数
        self.KARMAN: float = 0.4  # フォン・カルマン定数 (pp.210)
        self.weekday_key: str = "is_weekday"
        # 測定高度
        self.z_m: float = z_m

        # 図表の初期設定
        self.__setup_plot_params(labelsize, ticksize, plot_params)
        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = self.__setup_logger(logger, log_level)

    def __setup_logger(self, logger: Logger | None, log_level: int = INFO):
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
        logger: Logger = getLogger()
        logger.setLevel(log_level)  # ロガーのレベルを設定
        ch = StreamHandler()
        ch_formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(ch_formatter)  # フォーマッターをハンドラーに設定
        logger.addHandler(ch)  # StreamHandlerの追加
        return logger

    def __setup_plot_params(
        self, labelsize: float, ticksize: float, plot_params=None
    ) -> None:
        """
        matplotlibのプロットパラメータを設定します。

        引数:
            labelsize (float): 軸ラベルのフォントサイズ。
            ticksize (float): 軸目盛りのフォントサイズ。
            plot_params (Optional[Dict[str, any]]): matplotlibのプロットパラメータの辞書。
        """
        # デフォルトのプロットパラメータ
        default_params = {
            "font.family": "Arial",
            "axes.edgecolor": "None",
            "axes.labelcolor": "black",
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "grid.color": "gray",
            "axes.grid": False,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "ytick.direction": "out",
            "ytick.major.width": 1.0,
            "axes.linewidth": 1.0,
            "grid.linewidth": 1.0,
            "font.size": labelsize,
            "xtick.labelsize": ticksize,
            "ytick.labelsize": ticksize,
        }

        # plot_paramsが定義されている場合、デフォルトに追記
        if plot_params:
            default_params.update(plot_params)

        plt.rcParams.update(default_params)  # プロットパラメータを更新

    def __is_weekday(self, date: datetime) -> int:
        """
        指定された日付が平日であるかどうかを判定します。

        Args:
            date (datetime): 判定する日付。

        Returns:
            int: 平日であれば1、そうでなければ0。
        """
        return 1 if not jpholiday.is_holiday(date) and date.weekday() < 5 else 0

    def __prepare_csv(self, file_path: str) -> pd.DataFrame:
        """
        フラックスデータを含むCSVファイルを読み込み、処理します。

        引数:
            file_path (str): CSVファイルのパス。

        戻り値:
            pd.DataFrame: 処理済みのデータフレーム。
        """
        # CSVファイルの最初の行を読み込み、ヘッダーを取得するための一時データフレームを作成
        temp: pd.DataFrame = pd.read_csv(file_path, header=None, nrows=1, skiprows=0)
        header = temp.loc[temp.index[0]]

        # 実際のデータを読み込み、必要な行をスキップし、欠損値を指定
        df: pd.DataFrame = pd.read_csv(
            file_path,
            header=None,
            skiprows=2,
            na_values=["#DIV/0!", "#VALUE!", "#REF!", "#N/A", "#NAME?", "NAN"],
            low_memory=False,
        )
        # 取得したヘッダーをデータフレームに設定
        df.columns = header

        # self.required_columnsのカラムが存在するか確認
        missing_columns: list[str] = [
            col for col in self.required_columns if col not in df.columns.tolist()
        ]
        if missing_columns:
            raise ValueError(
                f"必要なカラムが不足しています: {', '.join(missing_columns)}"
            )

        # "Date"カラムをインデックスに設定して返却
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.dropna(subset=["Date"])
        df.set_index("Date", inplace=True)
        return df

    def combine_all_csv(self, csv_dir_path: str, suffix: str = ".csv") -> pd.DataFrame:
        """
        指定されたディレクトリ内の全CSVファイルを読み込み、処理し、結合します。
        Monthlyシートを結合することを想定しています。

        引数:
            csv_dir_path (str): CSVファイルが格納されているディレクトリのパス。
            suffix (str, optional): 読み込むファイルの拡張子。デフォルトは".csv"。

        戻り値:
            pandas.DataFrame: 結合および処理済みのデータフレーム。

        注意:
            - ディレクトリ内に少なくとも1つのCSVファイルが必要です。
        """
        csv_files = [f for f in os.listdir(csv_dir_path) if f.endswith(suffix)]
        if not csv_files:
            raise ValueError("指定されたディレクトリにCSVファイルが見つかりません。")

        df_array: list[pd.DataFrame] = []
        for csv_file in csv_files:
            file_path: str = os.path.join(csv_dir_path, csv_file)
            df: pd.DataFrame = self.__prepare_csv(file_path)
            df_array.append(df)

        # 結合
        df_combined: pd.DataFrame = pd.concat(df_array, join="outer")
        df_combined = df_combined.loc[~df_combined.index.duplicated(), :]

        # 平日と休日の判定に使用するカラムを作成
        df_combined[self.weekday_key] = df_combined.index.map(
            self.__is_weekday
        )  # 共通の関数を使用

        return df_combined

    def filter_data(
        self,
        df: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
        months: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        指定された期間や月でデータをフィルタリングするメソッド。

        Args:
            df (pd.DataFrame): フィルタリングするデータフレーム
            start_date (str | None): フィルタリングの開始日（'YYYY-MM-DD'形式）。デフォルトはNone。
            end_date (str | None): フィルタリングの終了日（'YYYY-MM-DD'形式）。デフォルトはNone。
            months (list[int] | None): フィルタリングする月のリスト（例：[1, 2, 12]）。デフォルトはNone。

        Returns:
            pd.DataFrame: フィルタリングされたデータフレーム
        """
        filtered_df: pd.DataFrame = df.copy()

        # 期間でフィルタリング
        if start_date is not None or end_date is not None:
            filtered_df = filtered_df.loc[start_date:end_date]

        # 月でフィルタリング
        if months is not None:
            filtered_df = filtered_df[filtered_df.index.month.isin(months)]

        return filtered_df

    def __source_area_KM2001(
        self,
        ksi: float,
        mu: float,
        dU: float,
        sigmaV: float,
        Z_d: float,
        max_ratio: float = 0.8,
    ) -> float:
        """
        Kormann and Meixner (2001)のフットプリントモデルに基づいてソースエリアを計算します。

        このメソッドは、与えられたパラメータを使用して、フラックスの80%寄与距離を計算します。
        計算は反復的に行われ、寄与率が80%に達するまで、または最大反復回数に達するまで続けられます。

        Args:
            ksi (float): フラックス長さスケール
            mu (float): 形状パラメータ
            dU (float): 風速の変化率
            sigmaV (float): 風速の標準偏差
            Z_d (float): ゼロ面変位高度
            max_ratio (float, optional): 寄与率の最大値。デフォルトは0.8。

        Returns:
            float: 80%寄与距離（メートル単位）。計算が収束しない場合はnp.nan。

        Note:
            - 計算が収束しない場合（最大反復回数に達した場合）、結果はnp.nanとなります。
        """
        if max_ratio > 1:
            raise ValueError("max_ratio は0以上1以下である必要があります。")
        # 変数の初期値
        sum_f: float = 0.0  # 寄与率(0 < sum_f < 1.0)
        x1: float = 0.0
        dF_xd: float = 0.0

        x_d: float = ksi / (
            1.0 + mu
        )  # Eq. 22 (x_d : クロスウィンド積分フラックスフットプリント最大位置)

        dx: float = x_d / 100.0  # 等値線の拡がりの最大距離の100分の1(m)

        # 寄与率が80%に達するまでfを積算
        while sum_f < (max_ratio / 1):
            x1 += dx

            # Equation 21 (dF : クロスウィンド積分フットプリント)
            dF: float = (
                pow(ksi, mu) * math.exp(-ksi / x1) / math.gamma(mu) / pow(x1, 1.0 + mu)
            )

            sum_f += dF  # Footprint を加えていく (0.0 < dF < 1.0)
            dx *= 2.0  # 距離は2倍ずつ増やしていく

            if dx > 1.0:
                dx = 1.0  # 一気に、1 m 以上はインクリメントしない
            if x1 > Z_d * 1000.0:
                break  # ソースエリアが測定高度の1000倍以上となった場合、エラーとして止める

        x_dst: float = x1  # 寄与率が80%に達するまでの積算距離
        f_last: float = (
            pow(ksi, mu)
            * math.exp(-ksi / x_dst)
            / math.gamma(mu)
            / pow(x_dst, 1.0 + mu)
        )  # Page 214 just below the Eq. 21.

        # y方向の最大距離とその位置のxの距離
        dy: float = x_d / 100.0  # 等値線の拡がりの最大距離の100分の1
        y_dst: float = 0.0
        accumulated_y: float = 0.0  # y方向の積算距離を表す変数

        # 最大反復回数を設定
        MAX_ITERATIONS: int = 100000
        for _ in range(MAX_ITERATIONS):
            accumulated_y += dy
            if accumulated_y >= x_dst:
                break

            dF_xd = (
                pow(ksi, mu)
                * math.exp(-ksi / accumulated_y)
                / math.gamma(mu)
                / pow(accumulated_y, 1.0 + mu)
            )  # 式21の直下（214ページ）

            aa: float = math.log(x_dst * dF_xd / f_last / accumulated_y)
            sigma: float = sigmaV * accumulated_y / dU  # 215ページ8行目

            if 2.0 * aa >= 0:
                y_dst_new: float = sigma * math.sqrt(2.0 * aa)
                if y_dst_new <= y_dst:
                    break  # forループを抜ける
                y_dst = y_dst_new

            dy = min(dy * 2.0, 1.0)

        else:
            # ループが正常に終了しなかった場合（最大反復回数に達した場合）
            x_dst = np.nan

        return x_dst

    def __calculate_ground_correction(self, data: pd.DataFrame) -> float:
        """
        地面修正量を計算します（Pennypacker and Baldocchi, 2016）。

        この関数は、与えられたデータフレームを使用して地面修正量を計算します。
        計算は以下のステップで行われます：
        1. 変位高さ（d）を計算
        2. 中立条件外のデータを除外
        3. 平均変位高さを計算
        4. 地面修正量を返す

        Args:
            data (pd.DataFrame): 風速や摩擦速度などのデータを含むDataFrame

        Returns:
            float: 計算された地面修正量
        """
        z: float = self.z_m
        # 変位高さ（d）の計算
        data["d"] = 0.6 * (
            z / (0.6 + 0.1 * (np.exp((self.KARMAN * data["WS vector"]) / data["u*"])))
        )

        # 中立条件外のデータを除外（中立条件：-0.1 < z/L < 0.1）
        data.loc[((data["z/L"] < -0.1) | (0.1 < data["z/L"])), "d"] = np.nan

        # 平均変位高さを計算
        d: float = data["d"].mean()

        # 地面修正量を返す
        return z - d

    def __calculate_stability_parameters(
        self, dzL: float
    ) -> tuple[float, float, float]:
        """
        安定性パラメータを計算します。

        大気安定度に基づいて、運動量とスカラーの安定度関数、および安定度パラメータを計算します。

        Args:
            dzL (float): 無次元高度 (z/L)、ここで z は測定高度、L はモニン・オブコフ長

        Returns:
            tuple[float, float, float]:
                phi_m (float): 運動量の安定度関数
                phi_c (float): スカラーの安定度関数
                n (float): 安定度パラメータ
        """
        phi_m: float = 0
        phi_c: float = 0
        n: float = 0
        if dzL > 0.0:
            # 安定成層の場合
            dzL = min(dzL, 2.0)
            phi_m = 1.0 + 5.0 * dzL
            phi_c = 1.0 + 5.0 * dzL
            n = 1.0 / (1.0 + 5.0 * dzL)
        else:
            # 不安定成層の場合
            phi_m = pow(1.0 - 16.0 * dzL, -0.25)
            phi_c = pow(1.0 - 16.0 * dzL, -0.50)
            n = (1.0 - 24.0 * dzL) / (1.0 - 16.0 * dzL)
        return phi_m, phi_c, n

    def __calculate_footprint_parameters(
        self, dUstar: float, dU: float, Z_d: float, phi_m: float, phi_c: float, n: float
    ) -> tuple[float, float, float, float, float]:
        """
        フットプリントパラメータを計算します。

        Args:
            dUstar (float): 摩擦速度
            dU (float): 風速
            Z_d (float): 地面修正後の測定高度
            phi_m (float): 運動量の安定度関数
            phi_c (float): スカラーの安定度関数
            n (float): 安定度パラメータ

        Returns:
            tuple[float, float, float, float, float]:
                m (べき指数),
                U (基準高度での風速),
                r (べき指数の補正項),
                mu (形状パラメータ),
                ksi (フラックス長さスケール)
        """
        KARMAN: float = self.KARMAN
        # パラメータの計算
        m: float = dUstar / KARMAN * phi_m / dU
        U: float = dU / pow(Z_d, m)
        r: float = 2.0 + m - n
        mu: float = (1.0 + m) / r
        kz: float = KARMAN * dUstar * Z_d / phi_c
        k: float = kz / pow(Z_d, n)
        ksi: float = U * pow(Z_d, r) / r / r / k
        return m, U, r, mu, ksi

    def __prepare_plot_data(
        self,
        x80: float,
        ksi: float,
        mu: float,
        r: float,
        U: float,
        m: float,
        sigmaV: float,
        flux_value: float,
        plot_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        フットプリントのプロットデータを準備します。

        Args:
            x80 (float): 80%寄与距離
            ksi (float): フラックス長さスケール
            mu (float): 形状パラメータ
            r (float): べき指数
            U (float): 風速
            m (float): 風速プロファイルのべき指数
            sigmaV (float): 風速の標準偏差
            flux_value (float): フラックス値
            plot_count (int): 生成するプロット数

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                x座標、y座標、フラックス値の配列のタプル
        """
        x_lim: int = int(x80)

        """
        各ランで生成するプロット数
        多いほどメモリに付加がかかるため注意
        """
        plot_num: int = plot_count  # 各ランで生成するプロット数

        # x方向の距離配列を生成
        x_list: np.ndarray = np.arange(1, x_lim + 1, dtype="float64")

        # クロスウィンド積分フットプリント関数を計算
        f_list: np.ndarray = (
            ksi**mu * np.exp(-ksi / x_list) / math.gamma(mu) / x_list ** (1.0 + mu)
        )

        # プロット数に基づいてx座標を生成
        num_list: np.ndarray = np.round(f_list * plot_num).astype("int64")
        x1: np.ndarray = np.repeat(x_list, num_list)

        # 風速プロファイルを計算
        Ux: np.ndarray = (
            (math.gamma(mu) / math.gamma(1 / r))
            * ((r**2 * self.KARMAN) / U) ** (m / r)
            * U
            * x1 ** (m / r)
        )

        # y方向の分散を計算し、正規分布に従ってy座標を生成
        sigma_array: np.ndarray = sigmaV * x1 / Ux
        y1: np.ndarray = np.random.normal(0, sigma_array)

        # フラックス値の配列を生成
        flux1 = np.full_like(x1, flux_value)

        return x1, y1, flux1

    def __rotate_coordinates(
        self, x: np.ndarray, y: np.ndarray, radian: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        座標を指定された角度で回転させます。

        この関数は、与えられたx座標とy座標を、指定された角度（ラジアン）で回転させます。
        回転は原点を中心に反時計回りに行われます。

        Args:
            x (np.ndarray): 回転させるx座標の配列
            y (np.ndarray): 回転させるy座標の配列
            radian (float): 回転角度（ラジアン）

        Returns:
            tuple[np.ndarray, np.ndarray]: 回転後の(x_, y_)座標の組
        """
        radian1: float = (radian - (np.pi / 2)) * (-1)
        x_: np.ndarray = x * np.cos(radian1) - y * np.sin(radian1)
        y_: np.ndarray = x * np.sin(radian1) + y * np.cos(radian1)
        return x_, y_

    def calculate_flux_footprint(
        self,
        df: pd.DataFrame,
        flux_key: str,
        plot_count: int = 50000,
        start_time: str = "10:00",
        end_time: str = "16:00",
    ) -> tuple[list[float], list[float], list[float]]:
        """
        フラックスフットプリントを計算し、指定された時間帯のデータを基に可視化します。

        引数:
            df (pd.DataFrame): 分析対象のデータフレーム。フラックスデータを含む。
            flux_key (str): フラックスデータの列名。計算に使用される。
            plot_count (int, optional): 生成するプロットの数。デフォルトは50000。
            start_time (str, optional): フットプリント計算に使用する開始時間。デフォルトは"10:00"。
            end_time (str, optional): フットプリント計算に使用する終了時間。デフォルトは"16:00"。

        戻り値:
            tuple[list[float], list[float], list[float]]:
                x座標 (メートル): タワーを原点とした東西方向の距離
                y座標 (メートル): タワーを原点とした南北方向の距離
                対象スカラー量の値: 各地点でのフラックス値

        注意:
            - 返却される座標は測定タワーを原点(0,0)とした相対位置です
            - すべての距離はメートル単位で表されます
            - 正のx値は東方向、正のy値は北方向を示します
        """
        df: pd.DataFrame = df.copy()  # 明示的にコピーを作成
        # データフレームのインデックスから日付の配列を作成
        datelist: np.ndarray = np.array(df.index.date)

        # 各日付が平日かどうかを判定し、リストに格納
        numbers: list[int] = [self.__is_weekday(date) for date in datelist]

        # weekday_keyに基づいてデータフレームに平日情報を追加
        df.loc[:, self.weekday_key] = numbers  # .locを使用して値を設定

        # 値が1のもの(平日)をコピーする
        data_weekday: pd.DataFrame = df[df[self.weekday_key] == 1].copy()
        # 特定の時間帯を抽出
        data_weekday = data_weekday.between_time(
            start_time, end_time
        )  # 引数を使用して時間帯を抽出
        data_weekday = data_weekday.dropna(subset=[flux_key])

        directions: list[float] = [
            wind_direction if wind_direction >= 0 else wind_direction + 360
            for wind_direction in data_weekday["Wind direction"]
        ]

        data_weekday.loc[:, "Wind direction_360"] = directions
        data_weekday.loc[:, "radian"] = data_weekday["Wind direction_360"] / 180 * np.pi

        # 風向が欠測なら除去
        data_weekday = data_weekday.dropna(subset=["Wind direction", flux_key])

        # 数値型への変換を確実に行う
        numeric_columns: list[str] = ["u*", "WS vector", "sigmaV", "z/L"]
        for col in numeric_columns:
            data_weekday[col] = pd.to_numeric(data_weekday[col], errors="coerce")

        # 地面修正量dの計算
        Z_d: float = self.__calculate_ground_correction(data_weekday)

        x_list: list[float] = []
        y_list: list[float] = []
        c_list: list[float] = []

        # tqdmを使用してプログレスバーを表示
        for i in tqdm(range(len(data_weekday)), desc="Calculating footprint"):
            dUstar: float = data_weekday["u*"].iloc[i]
            dU: float = data_weekday["WS vector"].iloc[i]
            sigmaV: float = data_weekday["sigmaV"].iloc[i]
            dzL: float = data_weekday["z/L"].iloc[i]

            if pd.isna(dUstar) or pd.isna(dU) or pd.isna(sigmaV) or pd.isna(dzL):
                self.logger.warn(f"#N/A fields are exist.: i = {i}")
                continue
            elif dUstar < 5.0 and dUstar != 0.0 and dU > 0.1:
                phi_m, phi_c, n = self.__calculate_stability_parameters(dzL)
                m, U, r, mu, ksi = self.__calculate_footprint_parameters(
                    dUstar, dU, Z_d, phi_m, phi_c, n
                )

                # ソースエリアの計算
                x80: float = self.__source_area_KM2001(ksi, mu, dU, sigmaV, Z_d)

                if not np.isnan(x80):
                    x1, y1, flux1 = self.__prepare_plot_data(
                        x80,
                        ksi,
                        mu,
                        r,
                        U,
                        m,
                        sigmaV,
                        data_weekday[flux_key].iloc[i],
                        plot_count=plot_count,
                    )
                    x1_, y1_ = self.__rotate_coordinates(
                        x1, y1, data_weekday["radian"].iloc[i]
                    )

                    x_list.extend(x1_)
                    y_list.extend(y1_)
                    c_list.extend(flux1)

        return (
            x_list,
            y_list,
            c_list,
        )

    def plot_flux_footprint(
        self,
        x_list: list[float],
        y_list: list[float],
        c_list: list[float],
        base_image_path: str,
        cmap: str,
        vmin: float,
        vmax: float,
        xy_min: float,
        xy_max: float,
        function: callable,
        cbar_label: str = "",
        cbar_labelpad: int = 20,
        output_path: str = "",
    ) -> plt.Figure | None:
        """
        フラックスフットプリントのプロットを作成します。

        Args:
            x_list (list[float]): プロットするデータのx座標リスト (メートル単位、タワーからの東西距離)
            y_list (list[float]): プロットするデータのy座標リスト (メートル単位、タワーからの南北距離)
            c_list (list[float]): プロットするデータの値リスト
            base_image_path (str): 使用する衛星画像ファイルのパス。
            cmap (str): カラーマップ名
            vmin (float): カラーマップの最小値
            vmax (float): カラーマップの最大値
            xy_min (float): プロットのx軸とy軸の最小値 (メートル)
            xy_max (float): プロットのx軸とy軸の最大値 (メートル)
            function (callable): データに適用する関数
            cbar_label (str, optional): カラーバーのラベル。デフォルトは空文字列
            cbar_labelpad (int, optional): カラーバーとラベルの間のpadding
            output_path (str, optional): 作成した図の保存先の絶対パス

        Returns:
            plt.Figure | None: 作成されたプロットのFigureオブジェクト、または何も返さない

        注意:
            - プロットの座標系は測定タワーを中心(0,0)とします
            - 背景の衛星画像は測定タワーが中心になるように配置されます
            - グリッドの目盛りはメートル単位です
        """
        if base_image_path == "":
            raise ValueError(f"base_image_path の値が不正です: '{base_image_path}'")
        base_image = Image.open(base_image_path)
        img_crop = base_image.crop((0, 0, 2160, 2160))

        plt.rcParams["axes.edgecolor"] = "None"
        fig: plt.Figure = plt.figure(figsize=(10, 8), dpi=300)  # figureのサイズを調整

        # メインの図の位置とサイズを調整
        ax_data: plt.Axes = fig.add_axes([0.05, 0.1, 0.8, 0.8])

        self.logger.info("プロットを作成中...")
        # フラックスの空間変動の可視化
        hexbin = ax_data.hexbin(
            x_list,
            y_list,
            c_list,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.3,
            gridsize=100,
            linewidths=0,
            mincnt=100,
            extent=[xy_min, xy_max, xy_min, xy_max],
            reduce_C_function=function,
        )

        ax_data.set_xlim(xy_min, xy_max)
        ax_data.set_xticks([])
        ax_data.set_ylim(xy_min, xy_max)
        ax_data.set_yticks([])
        ax_data.set_zorder(2)  # 前面に
        ax_data.patch.set_alpha(0)  # 背景を透明化
        ax_data.tick_params(axis="both", length=0, labelcolor="None")
        ax_data.grid(None)

        # 航空写真の画像の表示
        ax_img = ax_data.twiny().twinx()
        ax_img.imshow(img_crop, alpha=1, extent=[xy_min, xy_max, xy_max, xy_min])
        ax_img.set_xlim(xy_min, xy_max)
        ax_img.set_xticks([])
        ax_img.set_ylim(xy_max, xy_min)
        ax_img.set_yticks([])
        ax_img.tick_params(axis="both", length=0, labelcolor="None")
        ax_img.set_zorder(1)
        ax_img.grid(None)

        # カラーバーの追加（位置とサイズを調整）
        cbar_ax: plt.Axes = fig.add_axes(
            [0.88, 0.1, 0.03, 0.8]
        )  # カラーバーの位置を右側に調整
        cbar = fig.colorbar(hexbin, cax=cbar_ax)
        if cbar_label != "":
            cbar.set_label(
                cbar_label,
                rotation=270,
                labelpad=cbar_labelpad,
            )

        self.logger.info("プロットが正常に作成されました")

        # 出力パスが指定されている場合、図を保存
        if output_path != "":
            # 有効な拡張子のリスト
            valid_extensions: list[str] = [".png", ".jpg", ".jpeg", ".pdf", ".svg"]
            # ファイル名と拡張子を分離
            _, file_extension = os.path.splitext(output_path)

            # 拡張子が指定されていない、または有効でない場合
            if file_extension.lower() not in valid_extensions:
                # 拡張子にダブルクォーテーションをつけて出力
                quoted_extensions: list[str] = [f'"{ext}"' for ext in valid_extensions]
                self.logger.error(
                    f"`output_path`は有効な拡張子ではありません。プロットを保存するには、次のいずれかを指定してください: {','.join(quoted_extensions)}"
                )
                return
            else:
                try:
                    self.logger.info("プロットを保存中...")
                    fig.savefig(output_path)
                    self.logger.info(f"プロットが正常に保存されました: {output_path}")
                except Exception as e:
                    self.logger.error(
                        f"プロットの保存中にエラーが発生しました: {str(e)}"
                    )

        return fig
