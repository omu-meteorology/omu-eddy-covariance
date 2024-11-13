import os
from omu_eddy_covariance import FftFileReorganizer

# 変数定義
base_path = "/home/connect0459/workspace/labo/omu-eddy-covariance/workspace/transfer_function/private/2024.08.06"
input_dir_name: str = "fft"
output_dir_name: str = "sorted"
# input_dir_name: str = "fft-detrend"
# output_dir_name: str = "sorted-detrend"
flag_file_name: str = "Flg-202406211500_202408061100.csv"

# メイン処理
try:
    input_dir_path = os.path.join(base_path, input_dir_name)
    output_dir_path = os.path.join(base_path, output_dir_name)
    flag_file_path = os.path.join(base_path, flag_file_name)

    # インスタンスを作成
    reoirganizer = FftFileReorganizer(
        input_dir_path, output_dir_path, flag_file_path, sort_by_rh=False
    )
    reoirganizer.reorganize()
except KeyboardInterrupt:
    # キーボード割り込みが発生した場合、処理を中止する
    print("KeyboardInterrupt occurred. Abort processing.")
