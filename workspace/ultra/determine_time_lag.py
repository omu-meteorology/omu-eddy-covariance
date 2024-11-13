import os
import re
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt
from omu_eddy_covariance import EddyDataPreprocessor


def calculate_lags(key1: str, key2_list: list[str], df: pd.DataFrame) -> list[int]:
    """
    指定された基準変数（key1）と比較変数のリスト（key2_list）の間の遅れ時間（ラグ）を計算する。

    Args:
        key1 (str): 基準変数の列名。
        key2_list (list[str]): 比較変数の列名のリスト。
        df (pd.DataFrame): 遅れ時間の計算に使用するデータフレーム。

    Returns:
        list[int]: 各比較変数に対する遅れ時間（ラグ）のリスト。
    """
    lags_list: list[int] = []
    for _, key2 in enumerate(key2_list):
        # key1とkey2に一致するデータを取得
        data1: np.ndarray = np.array(df[key1].values)
        data2: np.ndarray = np.array(df[key2].values)

        """
        信号の平均値が0から離れている場合、相互相関関数の計算に影響を与えるので、
        sig1 = sig1 - sig1.mean()といったように平均を0にするといった処理を加える必要があります。
        """
        data1 = data1 - data1.mean()
        data2 = data2 - data2.mean()

        # データ点数
        data_length: int = len(data1)

        # 相互相関の計算
        correlation: np.ndarray = np.correlate(data1, data2, mode="full")

        # 相互相関のピークのインデックスを取得
        lag: int = int(correlation.argmax()) - (data_length - 1)

        lags_list.append(lag)
    return lags_list


def determine_time_lag(
    df: pd.DataFrame,
    median_range: float = 20,
    plot_range_tuple: tuple = (-200, 200),
    show_histogram: bool = False,
):
    """
    指定されたCSVファイルから遅れ時間（ラグ）を読み込み、ヒストグラムの表示、中央値、モード、平均値の計算を行う。

    Args:
        df (pandas.DataFrame): 遅れ時間（ラグ）が保存されているCSVファイルのパス。
        median_range (float, optional): 中央値を中心とした範囲。デフォルトは20。
        plot_range_tuple (tuple, optional): ヒストグラムの表示範囲。デフォルトは(-200, 200)。
        show_histogram (bool, optional): ヒストグラムを表示するかどうか。デフォルトはFalse。
    """
    # Read the lags CSV file
    lags_df: pd.DataFrame = df

    print(f"Calculate within the range of ±{median_range} from the median.")
    for column in lags_df.columns:
        data: pd.Series = lags_df[column]

        # Method 1: Histogram
        if show_histogram:
            plt.figure()
            plt.hist(data, bins=20, range=(plot_range_tuple))
            plt.title(f"Lags of {column}")
            plt.xlabel("Index")
            plt.ylabel("Frequency")
            plt.xlim(plot_range_tuple)
            plt.show()

        # Method 3: Median
        median_value = np.median(data)
        filtered_data = data[
            (data >= median_value - median_range)
            & (data <= median_value + median_range)
        ]

        # Method 4: Mode
        mode_result = stats.mode(filtered_data, keepdims=True)  # keepdimsをTrueに設定
        mode_value = mode_result.mode[0]

        # Method 5: Mean
        mean_value = np.mean(filtered_data)

        # print
        print(f'Results of "{column}".')
        print(f"Median: {median_value}")
        print(f"Mode  : {mode_value}")
        print(f"Mean  : {mean_value}")
        print("")


if __name__ == "__main__":
    target_home: str = "C:\\Users\\nakao\\workspace\\sac\\ultra\\2024.07.04\\Ultra_Eddy"
    # input_dir: str = f"{target_home}/eddy_csv"
    input_dir: str = f"{target_home}/eddy_csv-resampled"
    output_dir = f"{target_home}/eddy_lag"
    save_as_csv: bool = True

    # メイン処理
    # csv_files: list[str] = glob.glob(os.path.join(input_dir, "*.dat"), recursive=True)
    csv_files: list[str] = glob.glob(os.path.join(input_dir, "*.csv"), recursive=True)
    if not csv_files:
        print(f"There is no CSV file to process. Target directory: {input_dir}")

    # ファイル名に含まれる数字に基づいてソート
    csv_files = [f for f in csv_files if re.search(r"Eddy_(\d+)", f)]
    # 正規表現のマッチング結果がNoneでない場合のみ数値変換を行う
    csv_files.sort(
        key=lambda x: int(re.search(r"Eddy_(\d+)", x).group(1))
        if re.search(r"Eddy_(\d+)", x)
        else float("inf")
    )  # type: ignore

    key1: str = "wind_w"
    key2_list: list[str] = ["Tv", "Ultra_CH4_ppm_C", "Ultra_C2H6_ppb"]
    flag_once: bool = True
    all_lags: list[list[int]] = []
    try:
        for file in tqdm(csv_files, desc="Calculating"):
            if flag_once:
                tqdm.write("Delays: w-Tv, w-Ultra_CH4_ppm_C, w-Ultra_C2H6_ppb")
                flag_once = False
            path: str = os.path.join(input_dir, file)
            pre: EddyDataPreprocessor = EddyDataPreprocessor(path)
            df: pd.DataFrame = pre.execute(skiprows=0, drop_row=[])
            lags_list = calculate_lags(key1, key2_list, df)
            all_lags.append(lags_list)
            # tqdm.write(str(lags_list))
        print("Completed.")

        # Convert all_lags to a DataFrame
        lags_df = pd.DataFrame(all_lags, columns=key2_list)

        # Save lags_df to a CSV file
        if save_as_csv:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "lags.csv")
            lags_df.to_csv(output_file, index=False)
            print(f"Lags saved to {output_file}")

        # Calculate the median of each column
        median_values = {}
        for column in lags_df.columns:
            median_value = np.median(lags_df[column])
            median_values[column] = median_value

        # Print the median values
        print("Median values:")
        for column, median_value in median_values.items():
            print(f"{column}: {median_value}")

        # Determine time lag
        determine_time_lag(
            lags_df,
            median_range=20,
            # plot_range_tuple=(-100, 100),
            plot_range_tuple=(-150, 50),
            show_histogram=True,
        )

    except KeyboardInterrupt:
        print("Keyboard Interrupt occured.")
