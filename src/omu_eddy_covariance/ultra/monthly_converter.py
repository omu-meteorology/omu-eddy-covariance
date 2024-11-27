import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime


class MonthlyConverter:
    FILE_PATTERN = "SA.Ultra.*.xlsx"
    DATE_FORMAT = "%Y.%m"

    def __init__(self, directory: str | Path):
        """
        MonthlyConverterクラスのコンストラクタ

        Args:
            directory (str | Path): Excelファイルが格納されているディレクトリのパス
        """
        self.directory = Path(directory)
        if not self.directory.exists():
            raise NotADirectoryError(f"Directory not found: {self.directory}")

        # Excelファイルのパスを保持
        self.excel_files: dict[str, pd.ExcelFile] = {}

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
        return datetime.strptime(date_str, self.DATE_FORMAT)

    def _load_excel_files(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> None:
        """
        指定された日付範囲のExcelファイルを読み込む

        Args:
            start_date (str | None): 開始日 ('yyyy.MM'形式)
            end_date (str | None): 終了日 ('yyyy.MM'形式)
        """
        # 日付範囲の設定
        start_dt = (
            datetime.strptime(start_date, self.DATE_FORMAT) if start_date else None
        )
        end_dt = datetime.strptime(end_date, self.DATE_FORMAT) if end_date else None

        # 既存のファイルをクリア
        self.close()

        for excel_path in self.directory.glob(self.FILE_PATTERN):
            try:
                file_date = self._extract_date(excel_path.name)

                # 日付範囲チェック
                if start_dt and file_date < start_dt:
                    continue
                if end_dt and file_date > end_dt:
                    continue

                if excel_path.name not in self.excel_files:
                    self.excel_files[excel_path.name] = pd.ExcelFile(excel_path)

            except ValueError as e:
                print(f"Warning: Could not parse date from file {excel_path.name}: {e}")

    def read_sheets(
        self,
        sheet_names: str | list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sort_by_date: bool = True,
    ) -> pd.DataFrame:
        """
        指定されたシートを読み込み、DataFrameとして返却する
        2行目（単位の行）はスキップする

        Args:
            sheet_names (str | list[str]): 読み込むシート名。文字列または文字列のリスト
            start_date (str | None): 開始日 ('yyyy.MM'形式)
            end_date (str | None): 終了日 ('yyyy.MM'形式)
            sort_by_date (bool): ファイルの日付でソートするかどうか

        Returns:
            pd.DataFrame: 結合されたDataFrame
        """
        if isinstance(sheet_names, str):
            sheet_names = [sheet_names]

        # 指定された日付範囲のExcelファイルを読み込む
        self._load_excel_files(start_date, end_date)

        if not self.excel_files:
            raise ValueError("No Excel files found matching the criteria")

        dfs: list[pd.DataFrame] = []

        # ファイルを日付順にソート
        sorted_files = (
            sorted(self.excel_files.items(), key=lambda x: self._extract_date(x[0]))
            if sort_by_date
            else self.excel_files.items()
        )

        for file_name, excel_file in sorted_files:
            for sheet_name in sheet_names:
                if sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(
                        excel_file,
                        sheet_name=sheet_name,
                        header=0,
                        skiprows=[1],  # 2行目（単位の行）をスキップ
                    )
                    # ファイルの日付を列として追加
                    file_date = self._extract_date(file_name)
                    df["date"] = file_date
                    df["year"] = file_date.year
                    df["month"] = file_date.month
                    dfs.append(df)

        if not dfs:
            raise ValueError(f"No sheets found matching: {sheet_names}")

        return pd.concat(dfs, ignore_index=True)

    def get_available_dates(self) -> list[str]:
        """
        利用可能なファイルの日付一覧を返却する

        Returns:
            list[str]: 'yyyy.MM'形式の日付リスト
        """
        dates = []
        for file_name in self.directory.glob(self.FILE_PATTERN):
            try:
                date = self._extract_date(file_name.name)
                dates.append(date.strftime(self.DATE_FORMAT))
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
        if file_name not in self.excel_files:
            file_path = self.directory / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            self.excel_files[file_name] = pd.ExcelFile(file_path)
        return self.excel_files[file_name].sheet_names

    def close(self) -> None:
        """
        すべてのExcelファイルをクローズする
        """
        for excel_file in self.excel_files.values():
            excel_file.close()
        self.excel_files.clear()

    def __enter__(self) -> "MonthlyConverter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
