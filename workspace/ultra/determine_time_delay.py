from omu_eddy_covariance import EddyDataPreprocessor


if __name__ == "__main__":
    target_home: str = (
        "/home/z23641k/labo/omu-eddy-covariance/workspace/ultra/private/data"
    )
    input_dir: str = f"{target_home}/eddy_csv"
    output_dir: str = f"{target_home}/output"

    # メイン処理
    edp = EddyDataPreprocessor(fs=10)
    edp.analyze_time_delays(
        input_dir=input_dir,
        output_dir=output_dir,
        key1="wind_w",
        key2_list=["Tv", "Ultra_CH4_ppm_C", "Ultra_C2H6_ppb"],
    )
