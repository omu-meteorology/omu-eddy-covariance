import pandas as pd
import numpy as np
from pathlib import Path
from logging import getLogger, Formatter, Logger, StreamHandler, DEBUG, INFO


class UltraMonthlyAnalyzer:
    """Ultra分析計のデータを解析するクラス"""

    def __init__(
        self,
        data_dir: str = "src/data",
        output_dir: str = "src/out/csv",
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        Args:
            data_dir (str): Excelファイルが格納されているディレクトリパス
            output_dir (str): CSV出力先のディレクトリパス
            logger (Logger | None): 使用するロガー。Noneの場合は新しいロガーを生成します。
            logging_debug (bool): ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。
        """
        self.data_dir: Path = Path(data_dir)
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def load_monthly_data(
        self, year: int, month: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
        """
        指定された年月のExcelファイルを読み込む

        Parameters
        ----------
        year : int
            年
        month : int
            月

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, dict, dict]
            (Finalシートのデータ, Final.SAシートのデータ, Finalシートの単位情報, Final.SAシートの単位情報)
        """
        filename: str = f"SA.Ultra.{year}.{month:02d}.xlsx"
        filepath: Path = self.data_dir / filename

        self.logger.info(f"Loading {filename}")

        try:
            # ヘッダー行（1行目）の読み込み
            final_headers: pd.DataFrame = pd.read_excel(
                filepath, sheet_name="Final", nrows=1
            )
            final_sa_headers: pd.DataFrame = pd.read_excel(
                filepath, sheet_name="Final.SA", nrows=1
            )
            final_columns: pd.Index = final_headers.columns
            final_sa_columns: pd.Index = final_sa_headers.columns

            # 単位行（2行目）を直接読み込み
            # header=Noneを指定して列名を自動生成させない
            final_units_df: pd.DataFrame = pd.read_excel(
                filepath,
                sheet_name="Final",
                skiprows=1,  # 1行目をスキップ
                nrows=1,  # 1行（単位行）のみ読み込み
                header=None,  # 列名を自動生成させない
            )
            final_sa_units_df: pd.DataFrame = pd.read_excel(
                filepath, sheet_name="Final.SA", skiprows=1, nrows=1, header=None
            )

            # カラム名と単位を対応付け
            final_units_dict: dict = dict(
                zip(final_columns, final_units_df.iloc[0].values)
            )
            final_sa_units_dict: dict = dict(
                zip(final_sa_columns, final_sa_units_df.iloc[0].values)
            )

            # データの読み込み（2行目をスキップ）
            final_df: pd.DataFrame = pd.read_excel(
                filepath, sheet_name="Final", skiprows=[1]
            )
            final_sa_df: pd.DataFrame = pd.read_excel(
                filepath, sheet_name="Final.SA", skiprows=[1]
            )

            # カラム名を設定
            final_df.columns = final_columns
            final_sa_df.columns = final_sa_columns

            # 日付カラムをdatetime型に変換
            final_df["Date"] = pd.to_datetime(
                final_df["Date"], format="%Y/%m/%d %H:%M:%S"
            )
            final_sa_df["Date"] = pd.to_datetime(
                final_sa_df["Date"], format="%Y/%m/%d %H:%M:%S"
            )

            # 単位の文字列変換とnull値の処理
            for units_dict in [final_units_dict, final_sa_units_dict]:
                for key in units_dict:
                    if pd.isna(units_dict[key]) or units_dict[key] == "":
                        units_dict[key] = "--"
                    else:
                        # 文字列に変換して空白をアンダーバーに置換
                        units_dict[key] = str(units_dict[key]).strip().replace(" ", "_")

            # self.logger.info(
            #     f"Units found in Final sheet: CH4 ultra = {final_units_dict['CH4 ultra']}"
            # )
            # self.logger.info(f"All units from Final sheet: {final_units_dict}")

            return final_df, final_sa_df, final_units_dict, final_sa_units_dict

        except Exception as e:
            self.logger.error(f"Error loading Excel file: {str(e)}")
            raise

    def calculate_daily_average(self, year: int, month: int) -> None:
        """
        日平均を計算してCSVに出力する

        Parameters
        ----------
        year : int
            年
        month : int
            月
        """
        final_df, final_sa_df, final_units, final_sa_units = self.load_monthly_data(
            year, month
        )

        try:
            # Final シートの必要なカラムの定義
            final_columns: dict = {
                "CH4 ultra": "CH4_ultra",
                "C2H6 ultra": "C2H6_ultra",
                "Fch4 ultra": "Fch4_ultra",
                "Fc2h6 ultra": "Fc2h6_ultra",
            }

            # Final.SA シートの必要なカラムの定義
            sa_columns: dict = {
                "Wind direction": "Wind_direction",
                "CH4": "CH4_open",
                "Fch4": "Fch4_open",
            }

            # Final シートのデータ抽出と名前変更
            final_selected: pd.DataFrame = final_df[
                ["Date"] + list(final_columns.keys())
            ].copy()
            final_selected = final_selected.rename(
                columns={"Date": "Date", **final_columns}
            )

            # Final.SA シートのデータ抽出と名前変更
            final_sa_selected: pd.DataFrame = final_sa_df[
                ["Date"] + list(sa_columns.keys())
            ].copy()
            final_sa_selected = final_sa_selected.rename(
                columns={"Date": "Date", **sa_columns}
            )

            # 単位情報の取得と更新
            units: dict = {"Date": "--"}  # Date列の単位は常に '--'

            # Final シートの単位を取得
            for old_col, new_col in final_columns.items():
                units[new_col] = final_units[old_col]

            # Final.SA シートの単位を取得
            for old_col, new_col in sa_columns.items():
                units[new_col] = final_sa_units[old_col]

            # 日付のみを抽出して日平均を計算
            final_selected["Date"] = final_selected["Date"].dt.date
            final_sa_selected["Date"] = final_sa_selected["Date"].dt.date

            # エラー値を除外して日平均を計算
            daily_final: pd.DataFrame = final_selected.groupby("Date").agg(
                lambda x: x.replace(["#N/A", ""], np.nan)
                .astype(float)
                .mean(skipna=True)
            )
            daily_sa: pd.DataFrame = final_sa_selected.groupby("Date").agg(
                lambda x: x.replace(["#N/A", ""], np.nan)
                .astype(float)
                .mean(skipna=True)
            )

            # 結果を結合
            daily_average: pd.DataFrame = pd.merge(daily_final, daily_sa, on="Date")

            # CSVとして出力
            output_path: Path = (
                self.output_dir / f"daily_average-{year}_{month:02d}.csv"
            )

            # ヘッダー行を出力
            daily_average.to_csv(output_path, index=True)

            # ファイルを読み込んで単位行を追加
            with open(output_path, "r") as file:
                content: list[str] = file.readlines()

            # 単位行を作成（ヘッダーと同じ順序で）
            header_cols: list[str] = content[0].strip().split(",")
            units_line: str = (
                ",".join(units.get(col, "--") for col in header_cols) + "\n"
            )

            # 元のヘッダーと単位行を挿入
            with open(output_path, "w") as file:
                file.write(content[0])  # ヘッダー行
                file.write(units_line)  # 単位行
                file.writelines(content[1:])  # データ行

            self.logger.info(f"Daily average saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error in calculate_daily_average: {str(e)}")
            raise

    def calculate_diurnal_pattern(self, year: int, month: int) -> None:
        """
        日変化パターンを計算してCSVに出力する

        Parameters
        ----------
        year : int
            年
        month : int
            月
        """
        hour_col_key: str = "Hour"

        final_df, final_sa_df, final_units, final_sa_units = self.load_monthly_data(
            year, month
        )

        try:
            # 時刻のみを抽出し、新しい列として追加
            final_df = final_df.copy()
            final_sa_df = final_sa_df.copy()
            final_df[hour_col_key] = final_df["Date"].dt.hour
            final_sa_df[hour_col_key] = final_sa_df["Date"].dt.hour

            # フラックスデータの列名を特定
            final_flux_cols: list[str] = []
            sa_flux_cols: list[str] = []

            # Finalシートのフラックスカラムを確認
            for col in final_df.columns:
                if "Fch4 ultra" in col or "Fc2h6 ultra" in col:
                    final_flux_cols.append(col)

            # SAシートのフラックスカラムを確認
            for col in final_sa_df.columns:
                if col == "Fch4":  # SAシートの方は完全一致で検索
                    sa_flux_cols.append(col)

            # 単位情報を保存
            flux_units: dict = {hour_col_key: "--"}
            for col in final_flux_cols:
                flux_units[col] = final_units.get(col, "--")
            for col in sa_flux_cols:
                flux_units[col] = final_sa_units.get(col, "--")

            # フラックスデータと時間の抽出
            final_data: pd.DataFrame = final_df[[hour_col_key] + final_flux_cols].copy()
            sa_data: pd.DataFrame = final_sa_df[[hour_col_key] + sa_flux_cols].copy()

            # エラー値を除外して時間帯ごとの月平均を計算
            diurnal_final: pd.DataFrame = final_data.groupby(
                hour_col_key, as_index=False
            ).agg(
                lambda x: x.replace(["#N/A", ""], np.nan)
                .astype(float)
                .mean(skipna=True)
            )
            diurnal_sa: pd.DataFrame = sa_data.groupby(
                hour_col_key, as_index=False
            ).agg(
                lambda x: x.replace(["#N/A", ""], np.nan)
                .astype(float)
                .mean(skipna=True)
            )

            # 結果を結合
            diurnal_pattern: pd.DataFrame = pd.merge(
                diurnal_final, diurnal_sa, on=hour_col_key
            )

            # CSVとして出力
            output_path: Path = (
                self.output_dir / f"diurnal_fluxes-{year}_{month:02d}.csv"
            )

            # ヘッダー行を出力
            diurnal_pattern.to_csv(output_path, index=False)

            # 単位行を追加
            with open(output_path, "r") as file:
                content: list[str] = file.readlines()

            # 単位行を作成
            units_line: str = (
                "--,"
                + ",".join(
                    str(flux_units.get(col, "--"))
                    for col in diurnal_pattern.columns
                    if col != hour_col_key
                )
                + "\n"
            )

            # 元のヘッダーと単位行を挿入
            with open(output_path, "w") as file:
                file.write(content[0])  # ヘッダー行
                file.write(units_line)  # 単位行
                file.writelines(content[1:])  # データ行

            self.logger.info(f"Diurnal pattern saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error in calculate_diurnal_pattern: {str(e)}")
            raise
