import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from datetime import datetime
from omu_eddy_covariance import EddyDataPreprocessor


def resample_ultra_data(
    input_dir: str,
    resampled_dir: str,
    calc_py_dir: str,
    fs: float,
    output_resampled: bool = True,
    calculate_slopes: bool = True,
) -> None:
    """
    指定されたディレクトリ内のCSVファイルを処理し、リサンプリングと欠損値補間を行う。

    処理結果として、リサンプリングされたCSVファイルを出力し、相関係数やC2H6/CH4比を計算してDataFrameに保存する。

    Args:
        input_dir (str): 入力CSVファイルが格納されているディレクトリのパス
        resampled_dir (str): リサンプリングされたCSVファイルを出力するディレクトリのパス
        calc_py_dir (str): 計算結果を保存するディレクトリのパス
        fs (float): リサンプリングの周波数（Hz）
        output_resampled (bool, optional): リサンプリングされたCSVファイルを出力するかどうか。デフォルトはTrue
        calculate_slopes (bool, optional): 線形回帰をするかどうか。デフォルトはTrue

    Returns:
        None

    Raises:
        OSError: ディレクトリの作成に失敗した場合
        FileNotFoundError: 入力ファイルが見つからない場合
        ValueError: データの処理中にエラーが発生した場合
    """
    os.makedirs(resampled_dir, exist_ok=True)
    os.makedirs(calc_py_dir, exist_ok=True)

    ratio_data: list[dict[str, str | float]] = []
    latest_date: datetime = datetime.min

    csv_files: list[str] = [f for f in os.listdir(input_dir) if f.endswith(".dat")]
    csv_files.sort(
        key=lambda x: int(re.search(r"Eddy_(\d+)", x).group(1))
        if re.search(r"Eddy_(\d+)", x)
        else float("inf")
    )  # type: ignore

    for filename in tqdm(csv_files, desc="Processing files"):
        input_filepath: str = os.path.join(input_dir, filename)
        processor = EddyDataPreprocessor(input_filepath, fs, False)

        # リサンプリング＆欠損値補間
        df: pd.DataFrame = processor.execute()

        # 開始時間を取得
        start_time: datetime = df["TIMESTAMP"].iloc[0]
        # 処理したファイルの中で最も最新の日付
        latest_date = max(latest_date, start_time)

        # リサンプリング＆欠損値補間したCSVを出力
        if output_resampled:
            base_filename: str = re.sub(r"\.dat$", "", filename)
            output_csv_path: str = os.path.join(
                resampled_dir, f"{base_filename}-resampled.csv"
            )
            df.to_csv(output_csv_path, index=False)

        # 相関係数とC2H6/CH4比を計算
        if calculate_slopes:
            ch4_data: pd.Series = df["Ultra_CH4_ppm_C"]
            c2h6_data: pd.Series = df["Ultra_C2H6_ppb"]

            # 近似直線の傾き、切片、相関係数を計算
            ratio_row: dict[str, str | float] = {
                "Date": start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "Slope": f"{np.nan}",
                "Intercept": f"{np.nan}",
                "R": f"{np.nan}",
            }
            try:
                slope, intercept, r_value, _, _ = stats.linregress(ch4_data, c2h6_data)
                ratio_row = {
                    "Date": start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "Slope": f"{slope:.6f}",
                    "Intercept": f"{intercept:.6f}",
                    "R": f"{r_value:.6f}",
                }
            except Exception:
                # 何もせず、デフォルトの ratio_row を使用する
                pass

            # 結果をリストに追加
            ratio_data.append(ratio_row)

    if calculate_slopes:
        # DataFrameを作成し、Dateカラムで昇順ソート
        ratio_df: pd.DataFrame = pd.DataFrame(ratio_data)
        ratio_df["Date"] = pd.to_datetime(
            ratio_df["Date"]
        )  # Dateカラムをdatetime型に変換
        ratio_df = ratio_df.sort_values("Date")  # Dateカラムで昇順ソート

        # CSVとして保存
        ratio_filename: str = f"SAC.Ultra.{latest_date.strftime('%Y.%m.%d')}.ratio.csv"
        ratio_path: str = os.path.join(calc_py_dir, ratio_filename)
        ratio_df.to_csv(ratio_path, index=False)


if __name__ == "__main__":
    root_path: str = (
        "C:\\Users\\nakao\\workspace\\sac\\ultra\\data\\2024.11.01\\Ultra_Eddy"
    )

    input_dir: str = f"{root_path}\\eddy_csv"
    resampled_dir: str = f"{root_path}\\eddy_csv-resampled"
    calc_py_dir: str = f"{root_path}\\calc-py"
    fs: float = 10
    output_resampled: bool = True
    calculate_slopes: bool = True

    try:
        resample_ultra_data(
            input_dir,
            resampled_dir,
            calc_py_dir,
            fs,
            output_resampled,
            calculate_slopes,
        )
    except KeyboardInterrupt:
        # キーボード割り込みが発生した場合、処理を中止する
        print("KeyboardInterrupt occurred. Abort processing.")
