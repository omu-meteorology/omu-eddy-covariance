import os
import re
import csv
import shutil
from datetime import datetime
from logging import getLogger, Formatter, Logger, StreamHandler, DEBUG, INFO
from tqdm import tqdm


class FftFileReorganizer:
    """
    FFTファイルを再編成するためのクラス。

    入力ディレクトリからファイルを読み取り、フラグファイルに基づいて
    出力ディレクトリに再編成します。時間の完全一致を要求し、
    一致しないファイルはスキップして警告を出します。
    オプションで相対湿度（RH）に基づいたサブディレクトリへの分類も可能です。
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        flag_csv_path: str,
        sort_by_rh: bool = True,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        FftFileReorganizerクラスを初期化します。

        Args:
            input_path (str): 入力ファイルが格納されているディレクトリのパス
            output_path (str): 出力ファイルを格納するディレクトリのパス
            flag_csv_path (str): フラグ情報が記載されているCSVファイルのパス
            sort_by_rh (bool, optional): RHに基づいてサブディレクトリにファイルを分類するかどうか。デフォルトはTrue。
            logger (Logger | None): 使用するロガー。Noneの場合は新しいロガーを作成します。
            logging_debug (bool): ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。
        """
        self.fft_path: str = input_path
        self.sorted_path: str = output_path
        self.good_data_path: str = os.path.join(output_path, "good_data_all")
        self.bad_data_path: str = os.path.join(output_path, "bad_data")
        self.flag_file_path: str = flag_csv_path
        self.sort_by_rh: bool = sort_by_rh
        self.flags = {}
        self.warnings = []
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

    @staticmethod
    def parse_datetime(filename):
        """
        ファイル名から日時情報を抽出します。
        複数の異なるファイル名パターンに対応します。

        Args:
            filename (str): 解析対象のファイル名

        Returns:
            datetime: 抽出された日時情報

        Raises:
            ValueError: ファイル名から日時情報を抽出できない場合
        """
        """
        patterns (list[str]): 正規表現のパターン
        2024年8月現在はSACサイトのタワーおよびUltraに接続したロガーからの出力ファイルに対応
        """
        patterns: list[str] = [
            r"FFT_TOA5_\d+\.SAC_Eddy_\d+_(\d{4})_(\d{2})_(\d{2})_(\d{4})(?:\+)?\.csv",
            r"FFT_TOA5_\d+\.SAC_Ultra\.Eddy_\d+_(\d{4})_(\d{2})_(\d{2})_(\d{4})(?:\+)?(?:-resampled)?\.csv",
        ]

        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                year, month, day, time = match.groups()
                datetime_str: str = f"{year}{month}{day}{time}"
                return datetime.strptime(datetime_str, "%Y%m%d%H%M")

        raise ValueError(f"Could not parse datetime from filename: {filename}")

    def read_flag_file(self):
        """
        フラグファイルを読み込み、self.flagsディクショナリに格納します。
        """
        with open(self.flag_file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                time = datetime.strptime(row["time"], "%Y/%m/%d %H:%M")
                self.flags[time] = {"Flg": int(row["Flg"]), "RH": float(row["RH"])}

    def prepare_directories(self):
        """
        出力ディレクトリを準備します。
        既存のディレクトリがある場合は削除し、新しく作成します。
        """
        for path in [self.sorted_path, self.good_data_path, self.bad_data_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

        if self.sort_by_rh:
            for i in range(10, 101, 10):
                rh_path = os.path.join(self.sorted_path, f"RH{i}")
                os.makedirs(rh_path, exist_ok=True)

    def get_valid_files(self):
        """
        入力ディレクトリから有効なファイルのリストを取得します。

        Returns:
            list: 日時でソートされた有効なファイル名のリスト
        """
        fft_files = os.listdir(self.fft_path)
        valid_files = []
        for file in fft_files:
            try:
                self.parse_datetime(file)
                valid_files.append(file)
            except ValueError as e:
                self.warnings.append(f"{file} をスキップします: {str(e)}")
        return sorted(valid_files, key=self.parse_datetime)

    @staticmethod
    def get_rh_directory(rh: float):
        """
        RH値に基づいて適切なディレクトリ名を返します。

        Args:
            rh (float): 相対湿度値

        Returns:
            str: ディレクトリ名
        """
        if rh < 0 or rh > 100:  # 相対湿度として不正な値を除外
            return "bad_data"
        elif rh == 0:  # 0の場合はRH10に入れる
            return "RH10"
        elif rh.is_integer():  # int(整数)で表せる場合はその数を文字列に展開
            return f"RH{int(rh)}"
        else:  # 浮動小数の場合は整数に直す
            return f"RH{min(int(rh // 10 * 10 + 10), 100)}"

    def copy_files(self, valid_files):
        """
        有効なファイルを適切な出力ディレクトリにコピーします。
        フラグファイルの時間と完全に一致するファイルのみを処理します。

        Args:
            valid_files (list): コピーする有効なファイル名のリスト
        """
        with tqdm(total=len(valid_files)) as pbar:
            for filename in valid_files:
                src_file = os.path.join(self.fft_path, filename)
                file_time = self.parse_datetime(filename)

                if file_time in self.flags:
                    flag = self.flags[file_time]["Flg"]
                    rh = self.flags[file_time]["RH"]
                    if flag == 0:
                        # Copy to self.good_data_path
                        dst_file_good = os.path.join(self.good_data_path, filename)
                        shutil.copy2(src_file, dst_file_good)

                        if self.sort_by_rh:
                            # Copy to RH directory
                            rh_dir = self.get_rh_directory(rh)
                            dst_file_rh = os.path.join(
                                self.sorted_path, rh_dir, filename
                            )
                            shutil.copy2(src_file, dst_file_rh)
                    else:
                        dst_file = os.path.join(self.bad_data_path, filename)
                        shutil.copy2(src_file, dst_file)
                else:
                    self.warnings.append(
                        f"{filename} に対応するフラグが見つかりません。スキップします。"
                    )

                pbar.update(1)

    def reorganize(self):
        """
        ファイルの再編成プロセス全体を実行します。
        ディレクトリの準備、フラグファイルの読み込み、
        有効なファイルの取得、ファイルのコピーを順に行います。
        処理後、警告メッセージがあれば出力します。
        """
        self.prepare_directories()
        self.read_flag_file()
        valid_files = self.get_valid_files()
        self.copy_files(valid_files)
        self.logger.info("ファイルのコピーが完了しました。")

        if self.warnings:
            self.logger.warn("Warnings:")
            for warning in self.warnings:
                self.logger.warn(warning)
