import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
from logging import getLogger, Formatter, Logger, StreamHandler, DEBUG, INFO


class MonthlyConverter:
    """
    Monthlyシート（Excel）を一括で読み込み、DataFrameに変換するクラス。
    デフォルトは'SA.Ultra.*.xlsx'に対応していますが、コンストラクタのfile_patternを
    変更すると別のシートにも対応可能です（例: 'SA.Picaro.*.xlsx'）。
    """

    FILE_DATE_FORMAT = "%Y.%m"  # ファイル名用
    PERIOD_DATE_FORMAT = "%Y-%m-%d"  # 期間指定用

    def __init__(
        self,
        directory: str | Path,
        file_pattern: str = "SA.Ultra.*.xlsx",
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        MonthlyConverterクラスのコンストラクタ

        Args:
            directory (str | Path): Excelファイルが格納されているディレクトリのパス
            file_pattern (str): ファイル名のパターン。デフォルトは'SA.Ultra.*.xlsx'。
            logger (Logger | None): 使用するロガー。Noneの場合は新しいロガーを作成します。
            logging_debug (bool): ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。
        """
        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = MonthlyConverter.setup_logger(logger, log_level)

        self._directory = Path(directory)
        if not self._directory.exists():
            raise NotADirectoryError(f"Directory not found: {self._directory}")

        # Excelファイルのパスを保持
        self._excel_files: dict[str, pd.ExcelFile] = {}
        self._file_pattern: str = file_pattern

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

    def close(self) -> None:
        """
        すべてのExcelファイルをクローズする
        """
        for excel_file in self._excel_files.values():
            excel_file.close()
        self._excel_files.clear()

    def get_available_dates(self) -> list[str]:
        """
        利用可能なファイルの日付一覧を返却する

        Returns:
            list[str]: 'yyyy.MM'形式の日付リスト
        """
        dates = []
        for file_name in self._directory.glob(self._file_pattern):
            try:
                date = self._extract_date(file_name.name)
                dates.append(date.strftime(self.FILE_DATE_FORMAT))
            except ValueError:
                continue
        return sorted(dates)

    def get_sheet_names(self, file_name: str) -> list[str]:
        """
        指定されたファイルで利用可能なシート名の一覧を返却する

        Args:
            file_name (str): Excelファイル名

        Returns:
            list[str]: シート名のリスト
        """
        if file_name not in self._excel_files:
            file_path = self._directory / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            self._excel_files[file_name] = pd.ExcelFile(file_path)
        return self._excel_files[file_name].sheet_names

    def read_sheets(
        self,
        sheet_names: str | list[str],
        header: int = 0,
        skiprows: int | list[int] = [1],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_end_date: bool = True,
        sort_by_date: bool = True,
    ) -> pd.DataFrame:
        """
        指定されたシートを読み込み、DataFrameとして返却します。
        デフォルトでは2行目（単位の行）はスキップされます。

        Args:
            sheet_names (str | list[str]): 読み込むシート名。文字列または文字列のリストを指定できます。
            header (int): データのヘッダー行を指定します。デフォルトは0。
            skiprows (int | list[int]): スキップする行数。デフォルトでは1行目をスキップします。
            start_date (str | None): 開始日 ('yyyy-MM-dd')。この日付の'00:00:00'のデータが開始行となります。
            end_date (str | None): 終了日 ('yyyy-MM-dd')。この日付をデータに含めるかはinclude_end_dateフラグによって変わります。
            include_end_date (bool): 終了日を含めるかどうか。デフォルトはTrueです。
            sort_by_date (bool): ファイルの日付でソートするかどうか。デフォルトはTrueです。

        Returns:
            pd.DataFrame: 読み込まれたデータを結合したDataFrameを返します。
        """
        if isinstance(sheet_names, str):
            sheet_names = [sheet_names]

        # 指定された日付範囲のExcelファイルを読み込む
        self._load_excel_files(start_date, end_date)

        if not self._excel_files:
            raise ValueError("No Excel files found matching the criteria")

        dfs: list[pd.DataFrame] = []

        # ファイルを日付順にソート
        sorted_files = (
            sorted(self._excel_files.items(), key=lambda x: self._extract_date(x[0]))
            if sort_by_date
            else self._excel_files.items()
        )

        for file_name, excel_file in sorted_files:
            for sheet_name in sheet_names:
                if sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(
                        excel_file,
                        sheet_name=sheet_name,
                        header=header,
                        skiprows=skiprows,
                        na_values=[
                            "#DIV/0!",
                            "#VALUE!",
                            "#REF!",
                            "#N/A",
                            "#NAME?",
                            "NAN",
                        ],
                    )
                    # 月初日を含む完全な日付形式に変更
                    file_date = self._extract_date(file_name)
                    df["date"] = file_date.replace(day=1)  # 月の1日を設定
                    df["year"] = file_date.year
                    df["month"] = file_date.month
                    dfs.append(df)

        if not dfs:
            raise ValueError(f"No sheets found matching: {sheet_names}")

        combined_df = pd.concat(dfs, ignore_index=True)

        if start_date:
            start_dt = pd.to_datetime(start_date)
            combined_df = combined_df[combined_df["Date"] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            # 終了日を含む場合、翌日の0時を設定
            if include_end_date:
                end_dt += pd.Timedelta(days=1)
            combined_df = combined_df[
                combined_df["Date"] < end_dt
            ]  # 終了日の翌日0時より前のデータを全て含める

        return combined_df

    def __enter__(self) -> "MonthlyConverter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _extract_date(self, file_name: str) -> datetime:
        """
        ファイル名から日付を抽出する

        Args:
            file_name (str): "SA.Ultra.yyyy.MM.xlsx"形式のファイル名

        Returns:
            datetime: 抽出された日付
        """
        # ファイル名から日付部分を抽出
        date_str = ".".join(file_name.split(".")[-3:-1])  # "yyyy.MM"の部分を取得
        return datetime.strptime(date_str, self.FILE_DATE_FORMAT)

    def _load_excel_files(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> None:
        """
        指定された日付範囲のExcelファイルを読み込む

        Args:
            start_date (str | None): 開始日 ('yyyy-MM-dd'形式)
            end_date (str | None): 終了日 ('yyyy-MM-dd'形式)
        """
        # 期間指定がある場合は、yyyy-MM-dd形式から年月のみを抽出
        start_dt = None
        end_dt = None
        if start_date:
            temp_dt = datetime.strptime(start_date, self.PERIOD_DATE_FORMAT)
            start_dt = datetime(temp_dt.year, temp_dt.month, 1)
        if end_date:
            temp_dt = datetime.strptime(end_date, self.PERIOD_DATE_FORMAT)
            end_dt = datetime(temp_dt.year, temp_dt.month, 1)

        # 既存のファイルをクリア
        self.close()

        for excel_path in self._directory.glob(self._file_pattern):
            try:
                file_date = self._extract_date(excel_path.name)

                # 日付範囲チェック
                if start_dt and file_date < start_dt:
                    continue
                if end_dt and file_date > end_dt:
                    continue

                if excel_path.name not in self._excel_files:
                    self._excel_files[excel_path.name] = pd.ExcelFile(excel_path)

            except ValueError as e:
                self.logger.warning(
                    f"Could not parse date from file {excel_path.name}: {e}"
                )
