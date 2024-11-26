import numpy as np
import pandas as pd
from logging import getLogger, Formatter, Logger, StreamHandler, DEBUG, INFO


class EddyDataPreprocessor:
    def __init__(
        self,
        filepath: str,
        debug: bool = False,
        fs: float = 10,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        渦相関法によって記録されたデータファイルを処理するクラス

        Args:
            filepath (str): 読み込むCSVファイルのパス
            debug (bool): デバッグフラグ
            fs (float): サンプリング周波数
            logger (Logger | None): 使用するロガー。Noneの場合は新しいロガーを作成します。
            logging_debug (bool): ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。
        """
        self.filepath: str = filepath
        self.debug: bool = debug
        self.fs: float = fs

        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = EddyDataPreprocessor.setup_logger(logger, log_level)

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
        logger: Logger = getLogger()
        logger.setLevel(log_level)  # ロガーのレベルを設定
        ch = StreamHandler()
        ch_formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(ch_formatter)  # フォーマッターをハンドラーに設定
        logger.addHandler(ch)  # StreamHandlerの追加
        return logger

    def add_uvw_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrameに鉛直風速w、水平風速u、v の列を追加する関数

        Parameters:
            df (pd.DataFrame): 風速データを含むDataFrame

        Returns:
            df (pd.DataFrame): 鉛直風速w、水平風速u、vの列を追加したDataFrame
        """
        processed_df: pd.DataFrame = df.copy()
        # pandasの.valuesを使用してnumpy配列を取得し、その型をnp.ndarrayに明示的にキャストする
        wind_x_array: np.ndarray = np.array(processed_df["Ux"].values)
        wind_y_array: np.ndarray = np.array(processed_df["Uy"].values)
        wind_z_array: np.ndarray = np.array(processed_df["Uz"].values)

        # 平均風向を計算
        wind_direction: float = self.__wind_direction(wind_x_array, wind_y_array)

        # 水平方向に座標回転を行いu, v成分を求める
        wind_u_array, wind_v_array = self.__horizontal_wind_speed(
            wind_x_array, wind_y_array, wind_direction
        )
        wind_w_array: np.ndarray = wind_z_array  # wはz成分そのまま

        # u, wから風の迎角を計算
        wind_inclination: float = self.__wind_inclination(wind_u_array, wind_w_array)

        # 2回座標回転を行い、u, wを求める
        wind_u_array_rotated, wind_w_array_rotated = self.__vertical_rotation(
            wind_u_array, wind_w_array, wind_inclination
        )

        processed_df["wind_u"] = wind_u_array_rotated
        processed_df["wind_v"] = wind_v_array
        processed_df["wind_w"] = wind_w_array_rotated
        processed_df["rad_wind_dir"] = wind_direction
        processed_df["rad_wind_inc"] = wind_inclination
        processed_df["degree_wind_dir"] = np.degrees(wind_direction)
        processed_df["degree_wind_inc"] = np.degrees(wind_inclination)

        return processed_df

    def get_resampled_df(
        self,
        skiprows: int = 1,
        drop_row: list[int] | None = [3, 4],
        index_column: str = "TIMESTAMP",
        index_format: str = "%Y-%m-%d %H:%M:%S.%f",
        interpolate: bool = True,
        numeric_columns: list[str] = [
            "Ux",
            "Uy",
            "Uz",
            "Tv",
            "diag_sonic",
            "CO2_new",
            "H2O",
            "diag_irga",
            "cell_tmpr",
            "cell_press",
            "Ultra_CH4_ppm",
            "Ultra_C2H6_ppb",
            "Ultra_H2O_ppm",
            "Ultra_CH4_ppm_C",
            "Ultra_C2H6_ppb_C",
        ],
    ) -> pd.DataFrame:
        """
        CSVファイルを読み込み、前処理を行う

        前処理の手順は以下の通りです：
        1. 不要な行（インデックス0と1）を削除する
        2. 数値データを float 型に変換する
        3. TIMESTAMP列をDateTimeインデックスに設定する
        4. エラー値をNaNに置き換える
        5. 指定されたサンプリングレートでリサンプリングする
        6. 欠損値(NaN)を前後の値から線形補間する
        7. DateTimeインデックスを削除する

        Args:
            error_values (list[float], optional): エラー値のリスト。デフォルトは[-99999]。
            skiprows (int, optional): スキップする行数。デフォルトは1。
            index_column (str, optional): インデックスに使用する列名。デフォルトは'TIMESTAMP'。
            index_format (str, optional): インデックスの日付形式。デフォルトは'%Y-%m-%d %H:%M:%S.%f'。
            numeric_columns (list[str], optional): 数値型に変換する列名のリスト。
                デフォルトは["Ux", "Uy", "Uz", "Tv", "diag_sonic", "CO2_new", "H2O", "diag_irga", "cell_tmpr", "cell_press", "Ultra_CH4_ppm", "Ultra_C2H6_ppb", "Ultra_H2O_ppm", "Ultra_CH4_ppm_C", "Ultra_C2H6_ppb_C"]。

        Returns:
            df (pd.DataFrame): 前処理済みのデータフレーム
        """
        # CSVファイルを読み込む
        df: pd.DataFrame = pd.read_csv(self.filepath, skiprows=skiprows)

        if drop_row is not None and len(drop_row) > 0:
            # 削除するインデックスのリストに変換
            subtraction: int = skiprows + 2
            drop_index_list: list[int] = [row - subtraction for row in drop_row]
            # 不要な行を削除する
            df = df.drop(index=drop_index_list)

        # 数値データをfloat型に変換する
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

        # μ秒がない場合は".0"を追加する
        df[index_column] = df[index_column].apply(
            lambda x: f"{x}.0" if "." not in x else x
        )
        # TIMESTAMPをDateTimeインデックスに設定する
        df[index_column] = pd.to_datetime(df[index_column], format=index_format)
        # インデックスをセット
        df = df.set_index(index_column)

        # {self.fs}Hzでリサンプリングする
        resampling_period: int = int(1000 / self.fs)  # ms単位で算出
        # numeric_only=Trueは数値型のみ欠損補間を行う
        df_resampled: pd.DataFrame = df.resample(f"{resampling_period}ms").mean(
            numeric_only=True
        )

        # 欠損値をインターポレートで補完する
        if interpolate:
            df_resampled = df_resampled.interpolate()

        # DateTimeインデックスを削除する
        df = df_resampled.reset_index()
        return df

    def __horizontal_wind_speed(
        self, x_array: np.ndarray, y_array: np.ndarray, wind_dir: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        風速のu成分とv成分を計算する関数

        Parameters:
            x_array (numpy.ndarray): x方向の風速成分の配列
            y_array (numpy.ndarray): y方向の風速成分の配列
            wind_dir (float): 水平成分の風向（ラジアン）

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: u成分とv成分のタプル
        """
        # スカラー風速の計算
        scalar_hypotenuse: np.ndarray = np.sqrt(x_array**2 + y_array**2)

        instantaneous_wind_directions: np.ndarray = np.arctan2(y_array, x_array)
        # CSAT3では以下の補正が必要
        instantaneous_wind_directions = 0 - instantaneous_wind_directions

        # ベクトル風速の計算
        vector_u: np.ndarray = scalar_hypotenuse * np.cos(
            instantaneous_wind_directions - wind_dir
        )
        vector_v: np.ndarray = scalar_hypotenuse * np.sin(
            instantaneous_wind_directions - wind_dir
        )

        return vector_u, vector_v

    def __vertical_rotation(
        self,
        u_array: np.ndarray,
        w_array: np.ndarray,
        wind_inc: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        鉛直方向の座標回転を行い、u, wを求める関数

        Parameters:
            u_array (numpy.ndarray): u方向の風速
            w_array (numpy.ndarray): w方向の風速
            wind_inc (float): 平均風向に対する迎角（ラジアン）

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: 回転後のu, w
        """
        # 迎角を用いて鉛直方向に座標回転
        u_rotated = u_array * np.cos(wind_inc) + w_array * np.sin(wind_inc)
        w_rotated = w_array * np.cos(wind_inc) - u_array * np.sin(wind_inc)

        return u_rotated, w_rotated

    def __wind_direction(self, x_array: np.ndarray, y_array: np.ndarray) -> float:
        """
        水平方向の平均風向を計算する関数

        Parameters:
            x_array (numpy.ndarray): 東西方向の風速成分
            y_array (numpy.ndarray): 南北方向の風速成分

        Returns:
            wind_direction (float): 風向 (radians)
        """

        wind_direction: float = np.arctan2(np.mean(y_array), np.mean(x_array))
        # CSAT3では以下の補正が必要
        wind_direction = 0 - wind_direction
        return wind_direction

    def __wind_inclination(self, u_array: np.ndarray, w_array: np.ndarray) -> float:
        """
        平均風向に対する迎角を計算する関数

        Parameters:
            u_array (numpy.ndarray): u方向の瞬間風速
            w_array (numpy.ndarray): w方向の瞬間風速

        Returns:
            wind_inc (float): 平均風向に対する迎角（ラジアン）
        """
        wind_inc: float = np.arctan2(np.mean(w_array), np.mean(u_array))
        return wind_inc
