import os
from omu_eddy_covariance import TransferFunctionCalculator

# 変数定義
# base_path = r"C:\Users\nakao\workspace\sac\transfer_function\data\ultra\2024.10.07"
base_path = r"C:\Users\nakao\workspace\sac\ultra\data\2024.10.07\Ultra_Eddy\tf"
tf_file_name: str = "TF_Ultra.2024.10.07.csv"
# tf_file_name: str = "TF_Ultra.2024.10.07-detrend.csv"

show_cospectra_plot: bool = True
# show_cospectra_plot: bool = False
show_tf_plot: bool = True
# show_tf_plot: bool = False

# UltraのFFTファイルで使用されるキー名(スペース込み)
key_wt: str = "  f*cospec_wt/wt"
key_wch4: str = " f*cospec_wc/wc closed"
key_wc2h6: str = " f*cospec_wq/wq closed"

# メイン処理
try:
    file_path: str = os.path.join(base_path, tf_file_name)
    tfc = TransferFunctionCalculator(file_path, " f", 0.01, 1)

    # コスペクトルのプロット
    tfc.create_plot_cospectra(
        key1=key_wt,
        key2=key_wch4,
        label1=r"$fC_{wTv}$ / $\overline{w^\prime Tv^\prime}$",
        label2=r"$fC_{wCH_{4}}$ / $\overline{w^\prime CH_{4}^\prime}$",
        color2="red",
        subplot_label="(a)",
        show_plot=show_cospectra_plot,
    )

    tfc.create_plot_cospectra(
        key1=key_wt,
        key2=key_wc2h6,
        label1=r"$fC_{wTv}$ / $\overline{w^\prime Tv^\prime}$",
        label2=r"$fC_{wC_{2}H_{6}}$ / $\overline{w^\prime C_{2}H_{6}^\prime}$",
        color2="orange",
        subplot_label="(b)",
        show_plot=show_cospectra_plot,
    )

    print("伝達関数を分析中...")
    # 伝達関数の計算
    a_wch4, _, df_wch4 = tfc.calculate_transfer_function(
        reference_key=key_wt, target_key=key_wch4
    )
    a_wc2h6, _, df_wc2h6 = tfc.calculate_transfer_function(
        reference_key=key_wt, target_key=key_wc2h6
    )

    # カーブフィット図の作成
    tfc.create_plot_transfer_function(
        a=a_wch4,
        df_processed=df_wch4,
        reference_name="wTv",
        target_name="wCH4",
        show_plot=show_tf_plot,
    )
    tfc.create_plot_transfer_function(
        a=a_wc2h6,
        df_processed=df_wc2h6,
        reference_name="wTv",
        target_name="wC2H6",
        show_plot=show_tf_plot,
    )

    print(f"wCH4の係数 a: {a_wch4}")
    print(f"wC2H6の係数 a: {a_wc2h6}")
except KeyboardInterrupt:
    # キーボード割り込みが発生した場合、処理を中止する
    print("KeyboardInterrupt occurred. Abort processing.")
